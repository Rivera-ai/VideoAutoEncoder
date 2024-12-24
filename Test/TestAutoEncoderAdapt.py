from VideoAutoencoder import EfficientVideoAutoencoder, save_video_tensor, AdaptiveEfficientVideoAutoencoder
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import torchmetrics.functional as metrics
from itertools import cycle
from datasets import load_dataset

def collate_fn(batch):
    """Solo toma los videos del batch, ignorando el texto"""
    videos = [item[1] for item in batch]  # item[1] es el video tensor
    return torch.stack(videos)

class VideoProcessor:
    def __init__(self, target_size=(240, 426), fps=15, duration=2):
        self.target_size = target_size
        self.fps = fps
        self.duration = duration
        self.num_frames = fps * duration
        self.transform = self.normalize_video

    def normalize_video(self, video):
        """
        Normaliza y asegura las dimensiones correctas del video
        Input/Output: tensor de forma [C, T, H, W]
        """
        # Print shape for debugging
        #print(f"Input video shape: {video.shape}")
        
        # Ensure correct temporal dimension
        if video.shape[1] != self.num_frames:
            # Interpolate temporal dimension
            video = F.interpolate(
                video.unsqueeze(0),
                size=(self.num_frames, video.shape[2], video.shape[3]),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)
        
        # Ensure correct spatial dimensions
        if video.shape[2:] != self.target_size:
            # Interpolate spatial dimensions
            video = F.interpolate(
                video.unsqueeze(0),
                size=(video.shape[1], *self.target_size),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)
        
        # Normalize to [-1, 1]
        video = (video * 2) - 1
        
        # Print final shape for verification
        #print(f"Output video shape: {video.shape}")
        
        return video

    def process_batch(self, batch):
        """
        Process a batch of videos to ensure consistent dimensions
        """
        if isinstance(batch, torch.Tensor):
            # If batch is already stacked
            if batch.shape[-2:] != self.target_size or batch.shape[2] != self.num_frames:
                batch = F.interpolate(
                    batch,
                    size=(self.num_frames, *self.target_size),
                    mode='trilinear',
                    align_corners=False
                )
            return batch
        else:
            # If batch is a list
            processed = [self.normalize_video(video) for video in batch]
            return torch.stack(processed)

class VideoDataset(Dataset):
    def __init__(self, csv_path, video_folder, processor):
        self.video_folder = Path(video_folder)
        self.processor = processor
        self.data = pd.read_csv(csv_path)
        # Verificar existencia de archivos
        self.data = self.data[self.data.apply(
            lambda x: (self.video_folder / f"{x['video_name']}.mp4").exists(), 
            axis=1
        )]
        print(f"Dataset cargado con {len(self.data)} videos válidos")
    
    def __len__(self):
        return len(self.data)
    
    def load_video(self, video_path):
        try:
            import av
            with av.open(str(video_path)) as container:  # Usar context manager
                stream = container.streams.video[0]
            
            # Contar frames primero sin guardarlos
                total_frames = stream.frames
                if total_frames == 0:
                # Si no podemos obtener el conteo directamente, contamos manualmente
                    total_frames = sum(1 for _ in container.decode(video=0))
                    container.seek(0)  # Regresar al inicio
            

                if total_frames == 0:
                    raise ValueError("No frames found in video")
                
            # Calcular índices para muestreo uniforme
                if total_frames >= self.processor.num_frames:
                    indices = np.linspace(0, total_frames-1, self.processor.num_frames, dtype=int)
                else:
                    indices = np.arange(total_frames)
            
            # Crear tensor de salida
                video_tensor = torch.zeros(3, self.processor.num_frames, *self.processor.target_size)
            
            # Procesar frames uno por uno
                current_frame = 0
                for i, frame in enumerate(container.decode(video=0)):
                    if i not in indices:
                        continue
                    
                # Procesar solo los frames que necesitamos
                    output_idx = np.where(indices == i)[0][0]
                    if output_idx >= self.processor.num_frames:
                        break
                
                # Convertir frame a tensor directamente
                    img = frame.to_ndarray(format='rgb24')
                    img = torch.from_numpy(img).float() / 255.0
                
                # Redimensionar si es necesario
                    if img.shape[0:2] != self.processor.target_size:
                        img = F.interpolate(
                            img.permute(2, 0, 1).unsqueeze(0),
                            size=self.processor.target_size,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                    else:
                        img = img.permute(2, 0, 1)
                
                    video_tensor[:, output_idx] = img
                    current_frame = output_idx
            
            # Si faltan frames, repetir el último
                if current_frame + 1 < self.processor.num_frames:
                    video_tensor[:, current_frame+1:] = video_tensor[:, current_frame].unsqueeze(1).expand(
                        -1, self.processor.num_frames - (current_frame + 1), -1, -1
                    )
            
                return self.processor.transform(video_tensor)
            
        except Exception as e:
            print(f"Error cargando video {video_path}: {str(e)}")
            return torch.zeros(3, self.processor.num_frames, *self.processor.target_size)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_path = self.video_folder / f"{row['video_name']}.mp4"
        
        video_tensor = self.load_video(video_path)
        
        # Procesar texto
        text = row['answer']
        text_encoded = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
        text_tensor = torch.tensor(text_encoded, dtype=torch.long)
        
        return text_tensor, video_tensor

def TrainAutoEncoderBase():
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 10
    num_epochs = 1
    save_every = 5  # Guardar cada N pasos
    
    # Setup directorios
    results_folder = Path('./results')
    results_folder.mkdir(exist_ok=True, parents=True)
    
    # Dataset y modelo
    processor = VideoProcessor(target_size=(240, 426))  # Ajustado para 240p
    dataset = VideoDataset(
        csv_path="/teamspace/studios/this_studio/datos_videos.csv",
        video_folder="/teamspace/studios/this_studio/VideoDetailCaption/Test_Videos/",
        processor=processor
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )
    
    # Modelo y optimizador
    model = EfficientVideoAutoencoder(dim_latent=128).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * num_epochs)
    scaler = GradScaler()
    
    # Variables de tracking
    best_loss = float('inf')
    global_step = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for videos in pbar:
            try:
                videos = processor.process_batch(videos)
                print(f"Batch shape after processing: {videos.shape}")
                videos = videos.to(device)
                optimizer.zero_grad(set_to_none=True)
                
                with autocast():
                    reconstructed = model(videos)
                    print(f"Reconstructed shape: {reconstructed.shape}")
                    assert reconstructed.shape == videos.shape, \
                        f"Shape mismatch: reconstructed {reconstructed.shape} vs input {videos.shape}"
                    
                    # Pérdida combinada
                    recon_loss = F.l1_loss(reconstructed, videos)
                    ssim_loss = 1 - metrics.structural_similarity_index_measure(
                        reconstructed, 
                        videos,
                        data_range=2.0
                    )
                    loss = 0.7 * recon_loss + 0.3 * ssim_loss
                
                # Backward y optimize
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Logging
                pbar.set_description(
                    f'Epoch {epoch} | Loss: {loss.item():.4f} | '
                    f'Recon: {recon_loss.item():.4f} | SSIM: {ssim_loss.item():.4f}'
                )
                
                # Guardar checkpoints y muestras
                if global_step % save_every == 0:
                    # Guardar si es el mejor modelo
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_loss,
                        }, results_folder / 'best_model.pt')
                    
                    # Guardar muestra de reconstrucción
                    with torch.no_grad():                        
                        save_video_tensor(
                            reconstructed[0],
                            results_folder / f'recon_video_{global_step}.mp4',
                            fps=processor.fps
                        )
                
                global_step += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print("\nOOM Error, skipping batch")
                    continue
                raise e

def TrainAutoEncoderAdapt240p():
    """
    Entrena el AutoEncoder adaptativo específicamente para videos de 240p y 10 segundos
    """
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_epochs = 50
    save_every = 50
    
    # Configuración específica para 240p
    resolution = (240, 426)
    fps = 15
    duration = 5
    
    # Setup directorios
    results_folder = Path('./results_adaptive_240p')
    results_folder.mkdir(exist_ok=True, parents=True)
    
    # Inicializar procesador
    processor = VideoProcessor(
        target_size=resolution,
        fps=fps,
        duration=duration
    )
    
    # Dataset
    dataset = VideoDataset(
        csv_path="/teamspace/studios/this_studio/datos_videos.csv",
        video_folder="/teamspace/studios/this_studio/VideoDetailCaption/Test_Videos/",
        processor=processor
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )
    
    # Modelo y optimizador - Asegurando que todo esté en float32 inicialmente
    model = AdaptiveEfficientVideoAutoencoder(dim_latent=128, duration=5, quality='240p').to(device)
    model.print_model_info()
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * num_epochs)
    scaler = GradScaler()
    
    # Variables de tracking
    best_loss = float('inf')
    global_step = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} | 240p Training')
        
        for videos in pbar:
            try:
                # Procesar videos y moverlos a GPU en float32
                videos = processor.process_batch(videos)
                videos = videos.to(device, dtype=torch.float32)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Usar autocast para precisión mixta
                with autocast():
                    reconstructed = model(videos)
                    assert reconstructed.shape == videos.shape, \
                        f"Shape mismatch: reconstructed {reconstructed.shape} vs input {videos.shape}"
                    
                    # Pérdida combinada
                    recon_loss = F.l1_loss(reconstructed, videos)
                    ssim_loss = 1 - metrics.structural_similarity_index_measure(
                        reconstructed,
                        videos,
                        data_range=2.0
                    )
                    loss = 0.7 * recon_loss + 0.3 * ssim_loss
                
                # Backward y optimize con scaler
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Logging
                pbar.set_description(
                    f'Epoch {epoch} | 240p | Loss: {loss.item():.4f} | '
                    f'Recon: {recon_loss.item():.4f} | SSIM: {ssim_loss.item():.4f}'
                )
                
                # Guardar checkpoints y muestras
                if global_step % save_every == 0:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_loss,
                        }, results_folder / 'best_model_240p.pt')
                    
                    with torch.no_grad():
                        save_video_tensor(
                            reconstructed[0],
                            results_folder / f'recon_video_240p_{global_step}.mp4',
                            fps=processor.fps
                        )
                
                global_step += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print("\nOOM Error, skipping batch")
                    continue
                raise e

def TrainAutoEncoderAdapt480p():
    """
    Entrena el AutoEncoder adaptativo específicamente para videos de 480p y 5 segundos
    """
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1  # Batch más pequeño por mayor resolución
    num_epochs = 1
    save_every = 5
    
    # Configuración específica para 480p
    resolution = (480, 854)
    fps = 15
    duration = 5  # 5 segundos
    
    # Setup directorios
    results_folder = Path('./results_adaptive_480p')
    results_folder.mkdir(exist_ok=True, parents=True)
    
    # Inicializar procesador
    processor = VideoProcessor(
        target_size=resolution,
        fps=fps,
        duration=duration
    )
    
    # Dataset
    dataset = VideoDataset(
        csv_path="/teamspace/studios/this_studio/datos_videos.csv",
        video_folder="/teamspace/studios/this_studio/VideoDetailCaption/Test_Videos/",
        processor=processor
    )

    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )
    
    # Modelo y optimizador
    model = AdaptiveEfficientVideoAutoencoder(dim_latent=128, duration=5, quality='480p').to(device)
    model.print_model_info()
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * num_epochs)
    scaler = GradScaler()
    
    # Variables de tracking
    best_loss = float('inf')
    global_step = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} | 480p Training')
        
        for videos in pbar:
            try:
                videos = processor.process_batch(videos)
                videos = videos.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                
                with autocast():
                    reconstructed = model(videos)
                    assert reconstructed.shape == videos.shape, \
                        f"Shape mismatch: reconstructed {reconstructed.shape} vs input {videos.shape}"
                    
                    # Pérdida combinada
                    recon_loss = F.l1_loss(reconstructed, videos)
                    ssim_loss = 1 - metrics.structural_similarity_index_measure(
                        reconstructed,
                        videos,
                        data_range=2.0
                    )
                    loss = 0.8 * recon_loss + 0.2 * ssim_loss
                
                # Backward y optimize
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Logging
                pbar.set_description(
                    f'Epoch {epoch} | 480p | Loss: {loss.item():.4f} | '
                    f'Recon: {recon_loss.item():.4f} | SSIM: {ssim_loss.item():.4f}'
                )
                
                # Guardar checkpoints y muestras
                if global_step % save_every == 0:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_loss,
                        }, results_folder / 'best_model_480p.pt')
                    
                    with torch.no_grad():
                        save_video_tensor(
                            reconstructed[0],
                            results_folder / f'recon_video_480p_{global_step}.mp4',
                            fps=processor.fps
                        )
                
                global_step += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print("\nOOM Error, skipping batch")
                    continue
                raise e

if __name__ == '__main__':
#    print("Training AutoEncoder Base")
#    TrainAutoEncoderBase()
#    print("Training AutoEncoder 240p 5s")
#    TrainAutoEncoderAdapt240p()
    print("Training AutoEncoder 480p 5s")
    TrainAutoEncoderAdapt480p()