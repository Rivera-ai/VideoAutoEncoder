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
    def __init__(self, dim_latent=128, base_channels=32, min_spatial_size=15):
        super().__init__()
        self.dim_latent = dim_latent
        self.base_channels = base_channels
        self.min_spatial_size = min_spatial_size
        self._last_input_shape = None
        
        # Bloques de encoder y decoder se crearán dinámicamente
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        # Attention en el espacio latente
        self.attention = nn.Sequential(
            nn.Conv3d(dim_latent, dim_latent, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Capa de ajuste final
        self.final_adjust = nn.Conv3d(3, 3, kernel_size=3, padding=1)
        
    def _calculate_num_blocks(self, input_shape):
        """Calcula el número de bloques necesarios basado en las dimensiones de entrada"""
        _, _, T, H, W = input_shape
        
        # Calcula niveles necesarios para reducir la dimensión espacial hasta min_spatial_size
        spatial_levels = min(
            ceil(log2(H / self.min_spatial_size)),
            ceil(log2(W / self.min_spatial_size))
        )
        
        # Calcula niveles temporales (más conservador con la reducción temporal)
        temporal_levels = ceil(log2(T / self.min_spatial_size))
        
        return max(spatial_levels, temporal_levels, 2)  # Mínimo 2 niveles
        
    def _build_encoder(self, input_shape):
        """Construye dinámicamente los bloques del encoder"""
        self.encoder_blocks = nn.ModuleList()
        num_blocks = self._calculate_num_blocks(input_shape)
        current_channels = self.base_channels
        
        # Primer bloque (reducción espacial inicial)
        self.encoder_blocks.append(
            nn.Sequential(
                nn.Conv3d(3, current_channels, 
                         kernel_size=(3, 7, 7), 
                         stride=(1, 2, 2), 
                         padding=(1, 3, 3), 
                         bias=False),
                nn.BatchNorm3d(current_channels),
                nn.ReLU(inplace=True)
            )
        )
        
        # Bloques restantes
        for i in range(num_blocks - 1):
            next_channels = min(current_channels * 2, self.dim_latent)
            temporal_stride = 2 if i < 2 else 1  # Reducción temporal en primeros bloques
            
            self.encoder_blocks.append(
                EfficientResBlock3D(  # Cambiado a EfficientResBlock3D
                    current_channels, 
                    next_channels,
                    stride=2,
                    temporal_stride=temporal_stride
                )
            )
            current_channels = next_channels
            
    def _build_decoder(self, num_blocks, final_shape):
        """Construye dinámicamente los bloques del decoder"""
        self.decoder_blocks = nn.ModuleList()
        current_channels = self.dim_latent
        
        # Bloques de upsampling
        for i in range(num_blocks - 1):
            next_channels = max(current_channels // 2, self.base_channels)
            temporal_stride = 2 if i >= num_blocks - 3 else 1  # Upsampling temporal en últimos bloques
            
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        current_channels, 
                        next_channels,
                        kernel_size=4,
                        stride=(temporal_stride, 2, 2),
                        padding=1
                    ),
                    nn.BatchNorm3d(next_channels),
                    nn.ReLU(inplace=True)
                )
            )
            current_channels = next_channels
            
        # Bloque final
        self.decoder_blocks.append(
            nn.Sequential(
                nn.ConvTranspose3d(
                    current_channels,
                    3,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.Tanh()
            )
        )
        
    def _adapt_architecture(self, x):
        """Adapta la arquitectura basada en las dimensiones de entrada"""
        if not self.encoder_blocks or x.shape != self._last_input_shape:
            self._last_input_shape = x.shape
            num_blocks = self._calculate_num_blocks(x.shape)
            self._build_encoder(x.shape)
            self._build_decoder(num_blocks, x.shape)
            
            # Mover los bloques al mismo device y dtype que x
            self.to(x.device, x.dtype)
            
    def forward(self, x):
        # Adaptar arquitectura si es necesario
        self._adapt_architecture(x)
        
        # Encoding con gradient checkpointing
        h = x
        for block in self.encoder_blocks:
            h = checkpoint.checkpoint(block, h, use_reentrant=False)
        
        # Attention en espacio latente
        att = self.attention(h)
        h = h * att
        
        # Decoding con gradient checkpointing
        for block in self.decoder_blocks:
            h = checkpoint.checkpoint(block, h, use_reentrant=False)
        
        # Ajustar dimensiones finales si es necesario
        if h.shape[-3:] != x.shape[-3:]:
            h = F.interpolate(h, size=x.shape[-3:], mode='trilinear', align_corners=False)
        
        return self.final_adjust(h)