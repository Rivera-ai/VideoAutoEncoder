import torch
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