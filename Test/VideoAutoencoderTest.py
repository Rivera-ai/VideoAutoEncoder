from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision.utils import save_image
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.utils.checkpoint as checkpoint
from torchvision.utils import save_image
import torchmetrics.functional as metrics
import torchvision.io as io
import torchvision
import subprocess

def collate_fn(batch):
    """Solo toma los videos del batch, ignorando el texto"""
    videos = [item[1] for item in batch]  # item[1] es el video tensor
    return torch.stack(videos)

class EfficientVideoAutoencoder(nn.Module):
    def __init__(self, dim_latent=128):
        super().__init__()
        
        # Encoder más eficiente con downsampling
        self.encoder_blocks = nn.ModuleList([
            # Input: [B, 3, T=30, H=240, W=426]
            nn.Sequential(
                nn.Conv3d(3, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True)
            ),
            # Output: [B, 32, 30, 120, 213]
            
            EfficientResBlock3D(32, 64, stride=2, temporal_stride=2),
            # Output: [B, 64, 15, 60, 107]
            
            EfficientResBlock3D(64, 96, stride=2, temporal_stride=1),
            # Output: [B, 96, 15, 30, 54]
            
            EfficientResBlock3D(96, dim_latent, stride=2, temporal_stride=1)
            # Output: [B, 128, 15, 15, 27]
        ])
        
        # Attention en el espacio latente comprimido
        self.attention = nn.Sequential(
            nn.Conv3d(dim_latent, dim_latent, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Decoder con upsampling
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(dim_latent, 96, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(96),
                nn.ReLU(inplace=True)
            ),
            # Output: [B, 96, 15, 30, 54]
            
            nn.Sequential(
                nn.ConvTranspose3d(96, 64, kernel_size=4, stride=(1, 2, 2), padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True)
            ),
            # Output: [B, 64, 15, 60, 108]
            
            nn.Sequential(
                nn.ConvTranspose3d(64, 32, kernel_size=4, stride=(2, 2, 2), padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True)
            ),
            # Output: [B, 32, 30, 120, 216]
            
            nn.Sequential(
                nn.ConvTranspose3d(32, 3, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
            # Final output: [B, 3, 30, 240, 432]
            # Note: Might need final conv to adjust to exact dimensions
        ])
        
        # Capa final para ajustar dimensiones exactas si es necesario
        self.final_adjust = nn.Conv3d(3, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Print shape for debugging
        print(f"Input shape to model: {x.shape}")
        
        # Encoding con gradient checkpointing
        h = x
        for i, block in enumerate(self.encoder_blocks):
            h = checkpoint.checkpoint(block, h, use_reentrant=False)
            print(f"After encoder block {i}: {h.shape}")
        
        # Attention en espacio latente
        att = self.attention(h)
        h = h * att
        print(f"After attention: {h.shape}")
        
        # Decoding con gradient checkpointing
        for i, block in enumerate(self.decoder_blocks):
            h = checkpoint.checkpoint(block, h, use_reentrant=False)
            print(f"After decoder block {i}: {h.shape}")
        
        # Ajustar dimensiones finales si es necesario
        if h.shape[-2:] != x.shape[-2:]:
            h = F.interpolate(h, size=x.shape[-3:], mode='trilinear', align_corners=False)
        output = self.final_adjust(h)
        
        print(f"Final output shape: {output.shape}")
        return output

class EfficientResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, temporal_stride=1):
        super().__init__()
        if isinstance(stride, int):
            stride = (temporal_stride, stride, stride)
        
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if any(s != 1 for s in stride) or in_channels != out_channels:
            self.shortcut = nn.Conv3d(
                in_channels, 
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out, inplace=True)
        return out + self.shortcut(x)


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
        print(f"Input video shape: {video.shape}")
        
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
        print(f"Output video shape: {video.shape}")
        
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


import torch
import torchvision
import torchvision.io as io
import numpy as np
from pathlib import Path
import subprocess
from PIL import Image

def save_video_tensor(video_tensor, filepath, fps=30):
    """
    Guarda un tensor de video como archivo mp4 o frames individuales
    video_tensor: tensor de forma [T, C, H, W] o [C, T, H, W]
    """
    try:
        # Convertir a CPU, desconectar del grafo y normalizar
        video = video_tensor.cpu().detach().clamp(-1, 1)
        video = ((video + 1) / 2 * 255).to(torch.uint8)  # Convertir a uint8 aquí
        
        # Reorganizar dimensiones si es necesario
        if video.shape[0] == 3:  # Si está en formato [C, T, H, W]
            video = video.permute(1, 0, 2, 3)
        
        # Formato final: [T, H, W, C]
        video = video.permute(0, 2, 3, 1)
        
        filepath = Path(filepath)
        
        # Intentar primero con ffmpeg
        try:
            frames = video.numpy()
            save_video_ffmpeg(frames, str(filepath), fps)
            print(f"Video saved successfully to {filepath}")
            return
        except Exception as e:
            print(f"FFmpeg save failed: {e}, saving individual frames instead...")
            
            # Guardar frames individuales
            frames_dir = filepath.parent / f"{filepath.stem}_frames"
            frames_dir.mkdir(exist_ok=True, parents=True)
            
            # Guardar cada frame como PNG
            for i, frame in enumerate(video):
                frame_path = frames_dir / f"frame_{i:04d}.png"
                # Usar PIL para guardar la imagen
                Image.fromarray(frame.numpy()).save(str(frame_path))
            
            print(f"Saved {len(video)} frames to {frames_dir}")
            
            # Intentar combinar frames en video usando ffmpeg
            try:
                output_from_frames = filepath.parent / f"{filepath.stem}_from_frames.mp4"
                combine_frames_to_video(frames_dir, output_from_frames, fps)
                print(f"Combined frames into video: {output_from_frames}")
            except Exception as e:
                print(f"Failed to combine frames into video: {e}")
                
    except Exception as e:
        print(f"Error saving video: {e}")
        raise

def save_video_ffmpeg(frames, output_path, fps=30):
    """
    Guarda frames usando ffmpeg directamente
    frames: numpy array de forma [T, H, W, C]
    """
    if not isinstance(frames, np.ndarray):
        frames = frames.numpy()
        
    command = [
        'ffmpeg',
        '-y',  # Sobrescribir archivo si existe
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{frames.shape[2]}x{frames.shape[1]}',  # width x height
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',  # Input from pipe
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',  # Optimizar para streaming web
        output_path
    ]
    
    # Ejecutar ffmpeg
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        process.stdin.write(frames.tobytes())
        process.stdin.close()
        process.wait(timeout=30)
        
        if process.returncode != 0:
            stderr = process.stderr.read().decode()
            raise RuntimeError(f"FFmpeg failed with return code {process.returncode}: {stderr}")
    except Exception as e:
        raise RuntimeError(f"Error writing video with FFmpeg: {e}")
    finally:
        try:
            process.terminate()
        except:
            pass

def combine_frames_to_video(frames_dir, output_path, fps=30):
    """
    Combina frames individuales en un video usando ffmpeg
    """
    command = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        str(output_path)
    ]
    
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to combine frames: {process.stderr}")

        
def train():
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    num_epochs = 100
    save_every = 500  # Guardar cada N pasos
    
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
    
    # Modificar el __getitem__ para solo retornar el video
    original_getitem = dataset.__getitem__
    dataset.__getitem__ = lambda idx: original_getitem(idx)[1]
    
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
                        save_image(
                            reconstructed[0, :, 0].cpu(),
                            results_folder / f'recon_frame_{global_step}.png',
                            normalize=True
                        )
                        
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

if __name__ == "__main__":
    train()