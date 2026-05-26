import torch
import torch.nn as nn
import torch.nn.functional as F

class GuidedAttentionBlock(nn.Module):
    """
    Core O-SegNet Feature: Captures spatial-range dependencies between features
    at different scales to guide boundary semantic segmentation.
    
    Forces the network to correlate distinct pixel regions belonging to the subject,
    helping to outline fine boundaries (like fur texture and limb contours) 
    against complex forest and terrain backdrops.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Query, Key, and Value projections for self-attention mechanism
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable scale parameter initialized to zero as a residual shortcut multiplier
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, ch, h, w = x.size()
        
        # Project and reshape tensor matrices for batch matrix multiplication (BMM)
        proj_query = self.query(x).view(batch, -1, h * w).permute(0, 2, 1) # [B, N, C']
        proj_key   = self.key(x).view(batch, -1, h * w)                  # [B, C', N]
        
        # Generate spatial attention map reflecting contextual spatial dependencies
        energy = torch.bmm(proj_query, proj_key)                         # [B, N, N]
        attention = F.softmax(energy, dim=-1)
        
        # Multiply attention weights by value representations
        proj_value = self.value(x).view(batch, -1, h * w)                # [B, C, N]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))           # [B, C, N]
        
        # Re-project to the input spatial shape and add the residual connection
        out = out.view(batch, ch, h, w)
        return self.gamma * out + x


class PyramidPoolingNetwork(nn.Module):
    """
    O-SegNet Feature: Aggregates multi-scale global contextual information.
    
    Extracts features across distinct spatial bins (1x1, 2x2, 3x3, 6x6) to ensure 
    the model distinguishes large environmental background regions from localized 
    foreground biological features.
    """
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(size, size)),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            ) for size in pool_sizes
        ])
        
    def forward(self, x):
        h, w = x.size()[2:]
        # Extract low-dimensional feature maps at multiple scales, upscale them back to native size, and concatenate
        features = [x]
        for stage in self.stages:
            pooled_feat = stage(x)
            upsampled_feat = F.interpolate(pooled_feat, size=(h, w), mode='bilinear', align_corners=False)
            features.append(upsampled_feat)
            
        return torch.cat(features, dim=1)


class OSegNet(nn.Module):
    """
    The full custom O-SegNet structural framework.
    Maps input wildlife images to 2-class semantic segmentations (Background vs. Animal).
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # --- ENCODER STAGE ---
        # Block 1 (Handles basic edge extraction)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Downsizes by 2x
        )
        
        # Block 2 (Handles deep structural feature representations)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Downsizes by another 2x
        )
        
        # --- FEATURE FILTERING MIDDLEWARE ---
        # Guided Attention processing over deep feature vectors
        self.guided_attention = GuidedAttentionBlock(in_channels=128)
        
        # Multi-scale feature consolidation via Pyramid Pooling Network
        # Input: 128, Output: 128 + (32 * 4) = 256 channels
        self.ppn = PyramidPoolingNetwork(in_channels=128, pool_sizes=[1, 2, 3, 6])
        
        # --- DECODER STAGE ---
        # Channel reduction block
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final pixel-classification prediction layer (convolutional 1x1 mapping)
        self.final_seg_head = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Forward pass through feature extraction layers
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        
        # Apply guided feature attention alignment
        x2_att = self.guided_attention(x2)
        
        # Aggregate context through structural multi-bin pooling
        x_pooled = self.ppn(x2_att)
        
        # Decode features and upscale symmetrically to match original image dimensions
        d1 = self.dec1(x_pooled)
        d1_upsampled = F.interpolate(d1, scale_factor=4, mode='bilinear', align_corners=False)
        
        # Generate segmentation score matrices
        logits = self.final_seg_head(d1_upsampled)
        return logits


if __name__ == "__main__":
    # Structural verification check to confirm proper matrix transitions
    print("Executing structural verification test for custom OSegNet matrix dimensions...")
    test_tensor = torch.randn(2, 3, 224, 224) # Batch size: 2, Channels: 3, Dimensions: 224x224
    
    model = OSegNet(num_classes=2)
    output = model(test_tensor)
    
    print("\n--- MATRIX EVALUATION ---")
    print(f"Input Shape:  {test_tensor.shape}")
    print(f"Output Shape: {output.shape} (Matches expected format: [Batch, Classes, Height, Width])")
    
    assert output.shape == (2, 2, 224, 224), "Dimension validation failed."
    print("\n[Success] OSegNet compilation verified. Functional layers match proposal expectations.")