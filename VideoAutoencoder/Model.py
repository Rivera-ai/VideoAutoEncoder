import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

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