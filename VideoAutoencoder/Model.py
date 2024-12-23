import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from math import log2, ceil

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

class AdaptiveEfficientVideoAutoencoder(nn.Module):
    """
Adaptive autoencoder for video compression with varying qualities and durations.

This model implements an autoencoder that automatically adapts to different:
- Video resolutions (240p, 360p, 480p, 720p)
- Video durations
- Frame rates (FPS)

Key features:
- Adaptive architecture that adjusts channels based on video quality
- Intelligent temporal reduction system based on duration
- Skip connections with automatic dimension adjustment
- Attention mechanism in the latent space
- Gradient checkpointing for memory optimization

Example usage:
```python
# Create model for 240p videos, 5 seconds at 15 FPS
model = AdaptiveVideoAutoencoder(
    dim_latent=128,
    fps=15,
    duration=5,
    quality='240p'
)

# Process a batch of videos
# videos.shape = [batch_size, channels=3, frames=75, height=240, width=426]
reconstructed = model(videos)

# Print model information
model.print_model_info()
"""
    def __init__(self, 
                 dim_latent=128,
                 fps=15,
                 duration=5,
                 quality='240p'):
        super().__init__()
        
        # Configuración base
        self.dim_latent = dim_latent
        self.fps = fps
        self.duration = duration
        self.quality = quality
        
        # Configuraciones de calidad
        self.quality_configs = {
            '240p': {'height': 240, 'width': 426, 'channels_mult': 1.0},
            '360p': {'height': 360, 'width': 640, 'channels_mult': 1.5},
            '480p': {'height': 480, 'width': 854, 'channels_mult': 2.0},
            '720p': {'height': 720, 'width': 1280, 'channels_mult': 3.0}
        }
        
        # Obtener multiplicador de canales según calidad
        self.channels_mult = self.quality_configs[quality]['channels_mult']
        
        # Calcular canales base ajustados por calidad
        self.channels = {
            'c1': int(32 * self.channels_mult),
            'c2': int(64 * self.channels_mult),
            'c3': int(96 * self.channels_mult),
            'c4': dim_latent
        }
        
        # Calcular reducción temporal según duración
        self.temporal_reduction = self._calculate_temporal_reduction()
        
        # Construir encoder
        self.encoder_blocks = nn.ModuleList([
            # Bloque inicial
            nn.Sequential(
                nn.Conv3d(3, self.channels['c1'], 
                         kernel_size=(3, 7, 7),
                         stride=(self.temporal_reduction[0], 2, 2),
                         padding=(1, 3, 3),
                         bias=False),
                nn.BatchNorm3d(self.channels['c1']),
                nn.ReLU(inplace=True)
            ),
            
            # Bloques residuales
            EfficientResBlock3D(
                self.channels['c1'], 
                self.channels['c2'],
                stride=2,
                temporal_stride=self.temporal_reduction[1]
            ),
            
            EfficientResBlock3D(
                self.channels['c2'],
                self.channels['c3'],
                stride=2,
                temporal_stride=self.temporal_reduction[2]
            ),
            
            EfficientResBlock3D(
                self.channels['c3'],
                self.channels['c4'],
                stride=2,
                temporal_stride=self.temporal_reduction[3]
            )
        ])
        
        # Mecanismo de atención mejorado
        self.attention = nn.Sequential(
            nn.Conv3d(dim_latent, dim_latent, kernel_size=1),
            nn.BatchNorm3d(dim_latent),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim_latent, dim_latent, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Decoder con skip connections
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(dim_latent, self.channels['c3'],
                                 kernel_size=4,
                                 stride=(self.temporal_reduction[3], 2, 2),
                                 padding=1),
                nn.BatchNorm3d(self.channels['c3']),
                nn.ReLU(inplace=True)
            ),
            
            nn.Sequential(
                nn.ConvTranspose3d(self.channels['c3'], self.channels['c2'],
                                 kernel_size=4,
                                 stride=(self.temporal_reduction[2], 2, 2),
                                 padding=1),
                nn.BatchNorm3d(self.channels['c2']),
                nn.ReLU(inplace=True)
            ),
            
            nn.Sequential(
                nn.ConvTranspose3d(self.channels['c2'], self.channels['c1'],
                                 kernel_size=4,
                                 stride=(self.temporal_reduction[1], 2, 2),
                                 padding=1),
                nn.BatchNorm3d(self.channels['c1']),
                nn.ReLU(inplace=True)
            ),
            
            nn.Sequential(
                nn.ConvTranspose3d(self.channels['c1'], 3,
                                 kernel_size=4,
                                 stride=(self.temporal_reduction[0], 2, 2),
                                 padding=1),
                nn.Tanh()
            )
        ])
        
        # Skip connections adjustments
        self.skip_adjustments = nn.ModuleList([
            nn.Conv3d(self.channels['c3'], self.channels['c3'], kernel_size=1),
            nn.Conv3d(self.channels['c2'], self.channels['c2'], kernel_size=1),
            nn.Conv3d(self.channels['c1'], self.channels['c1'], kernel_size=1)
        ])
        
        # Ajuste final para dimensiones exactas
        self.final_adjust = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def _calculate_temporal_reduction(self):
        """Calcula la estrategia de reducción temporal basada en la duración"""
        if self.duration >= 10:
            return [2, 2, 2, 1]  # Reducción agresiva
        elif self.duration >= 5:
            return [1, 2, 1, 1]  # Reducción moderada
        else:
            return [1, 1, 1, 1]  # Sin reducción temporal
    
    def forward(self, x):
        print(f"Input shape to model: {x.shape}")
        
        # Encoding con almacenamiento de features
        h = x
        features = []
        
        for i, block in enumerate(self.encoder_blocks):
            h = checkpoint.checkpoint(block, h, use_reentrant=False)
            print(f"Encoder block {i} output shape: {h.shape}")
            features.append(h)
        
        # Attention con conexión residual
        att = self.attention(h)
        h = h * att + h
        print(f"After attention shape: {h.shape}")
        
        # Decoding con skip connections
        for i, block in enumerate(self.decoder_blocks[:-1]):
            h = checkpoint.checkpoint(block, h, use_reentrant=False)
            if i < len(self.skip_adjustments):
                skip = self.skip_adjustments[i](features[-(i+2)])
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], 
                                   mode='trilinear', align_corners=False)
                h = h + skip
            print(f"Decoder block {i} output shape: {h.shape}")
        
        # Bloque final y ajuste de dimensiones
        h = self.decoder_blocks[-1](h)
        if h.shape[-3:] != x.shape[-3:]:
            h = F.interpolate(h, size=x.shape[-3:], 
                            mode='trilinear', align_corners=False)
        
        output = self.final_adjust(h)
        print(f"Final output shape: {output.shape}")
        return output
    
    def print_model_info(self):
        """Imprime información sobre la configuración del modelo"""
        print(f"\nModelo configurado para:")
        print(f"- Calidad: {self.quality}")
        print(f"- FPS: {self.fps}")
        print(f"- Duración: {self.duration}s")
        print(f"- Frames totales: {self.fps * self.duration}")
        print(f"- Multiplicador de canales: {self.channels_mult}")
        print(f"- Canales: {self.channels}")
        print(f"- Dimensión latente: {self.dim_latent}")
        quality_config = self.quality_configs[self.quality]
        print(f"- Resolución: {quality_config['height']}x{quality_config['width']}")
        print(f"- Reducción temporal: {self.temporal_reduction}")