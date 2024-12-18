import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class EfficientVideoAutoencoder(nn.Module):
    def __init__(self, dim_latent=128):
        super().__init__()
        
        # Encoder más eficiente
        self.encoder_blocks = nn.ModuleList([
            # Input: [B, 3, T=30, H=240, W=426]
            nn.Sequential(
                nn.Conv3d(3, 32, kernel_size=(3, 7, 7), stride=(1, 1, 1), padding=(1, 3, 3), bias=False),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True)
            ),
            # Output: [B, 32, 30, 240, 426]
            
            EfficientResBlock3D(32, 64, stride=1, temporal_stride=1),
            # Output: [B, 64, 30, 240, 426]
            
            EfficientResBlock3D(64, 96, stride=1, temporal_stride=1),
            # Output: [B, 96, 30, 240, 426]
            
            EfficientResBlock3D(96, dim_latent, stride=1, temporal_stride=1)
            # Output: [B, 128, 30, 240, 426]
        ])
        
        # Attention ligero
        self.attention = nn.Sequential(
            nn.Conv3d(dim_latent, dim_latent, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Decoder más simple
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(dim_latent, 96, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(96),
                nn.ReLU(inplace=True)
            ),
            # Output: [B, 96, 30, 240, 426]
            
            nn.Sequential(
                nn.Conv3d(96, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True)
            ),
            # Output: [B, 64, 30, 240, 426]
            
            nn.Sequential(
                nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True)
            ),
            # Output: [B, 32, 30, 240, 426]
            
            nn.Sequential(
                nn.Conv3d(32, 3, kernel_size=3, padding=1),
                nn.Tanh()
            )
            # Final output: [B, 3, 30, 240, 426]
        ])
    
    def forward(self, x):
        # Print shape for debugging
        print(f"Input shape to model: {x.shape}")
        
        # Encoding con gradient checkpointing
        h = x
        for i, block in enumerate(self.encoder_blocks):
            h = checkpoint.checkpoint(block, h, use_reentrant=False)
            print(f"After encoder block {i}: {h.shape}")
        
        # Attention ligero
        att = self.attention(h)
        h = h * att
        print(f"After attention: {h.shape}")
        
        # Decoding con gradient checkpointing
        for i, block in enumerate(self.decoder_blocks[:-1]):
            h = checkpoint.checkpoint(block, h, use_reentrant=False)
            print(f"After decoder block {i}: {h.shape}")
        
        # Última capa sin checkpoint
        output = self.decoder_blocks[-1](h)
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