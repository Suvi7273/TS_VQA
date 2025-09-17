import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Backbone, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove fully connected layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels  
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c2 = self.layer1(x)  # 1/4 resolution
        c3 = self.layer2(c2) # 1/8 resolution
        c4 = self.layer3(c3) # 1/16 resolution
        c5 = self.layer4(c4) # 1/32 resolution
        
        return [c2, c3, c4, c5]

class ReceptiveEnhancementModule(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, kernel_size=7):
        super(ReceptiveEnhancementModule, self).__init__()
        
        # Large kernel convolution for receptive field enhancement
        self.large_conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=2, 
            padding=kernel_size//2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # x is the highest level feature from ResNet50 (c5)
        x = self.large_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class FeatureTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=2048):
        super(FeatureTransformerEncoder, self).__init__()
        
        # Feature projection layers
        self.input_proj_c2 = nn.Conv2d(256, d_model, kernel_size=1)
        self.input_proj_c3 = nn.Conv2d(512, d_model, kernel_size=1)  
        self.input_proj_c4 = nn.Conv2d(1024, d_model, kernel_size=1)
        self.input_proj_c5 = nn.Conv2d(2048, d_model, kernel_size=1)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, d_model, 100, 100))
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        self.channel_reduction = nn.Conv2d(5 * d_model, d_model, kernel_size=1)

        
    def forward(self, features, rem_features):
        # features: [c2, c3, c4, c5] from ResNet50
        c2, c3, c4, c5 = features
        
        # Project all features to same dimension
        c2_proj = self.input_proj_c2(c2)
        c3_proj = self.input_proj_c3(c3)
        c4_proj = self.input_proj_c4(c4)
        c5_proj = self.input_proj_c5(c5)
        
        # Resize to same spatial dimensions (use c4 as reference)
        target_size = c4_proj.shape[-2:]
        
        c2_resized = F.interpolate(c2_proj, size=target_size, mode='bilinear', align_corners=False)
        c3_resized = F.interpolate(c3_proj, size=target_size, mode='bilinear', align_corners=False)
        c5_resized = F.interpolate(c5_proj, size=target_size, mode='bilinear', align_corners=False)
        rem_resized = F.interpolate(rem_features, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate features
        multi_scale_features = torch.cat(
            [c2_resized, c3_resized, c4_proj, c5_resized, rem_resized], dim=1
        )  # [B, 5*d_model, H, W]
        
        # Important: define Conv2d in __init__, not inside forward
        multi_scale_features = self.channel_reduction(multi_scale_features)
        
        # Add positional encoding
        B, C, H, W = multi_scale_features.shape[0]
        pos_embed = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        multi_scale_features = multi_scale_features + pos_embed
        
        # Flatten spatial dimensions for transformer
        features_flat = multi_scale_features.flatten(2).permute(2, 0, 1)
        
        # Apply transformer encoder
        enhanced_features = self.transformer_encoder(features_flat)
        
        # Reshape back to [B, C, H, W]
        enhanced_features = enhanced_features.permute(1, 2, 0).view(B, C, H, W)
        
        return enhanced_features


class VimTSFeatureExtraction(nn.Module):
    def __init__(self, pretrained=True):
        super(VimTSFeatureExtraction, self).__init__()
        
        # Components
        self.resnet_backbone = ResNet50Backbone(pretrained=pretrained)
        self.rem = ReceptiveEnhancementModule(in_channels=2048, out_channels=256)
        self.transformer_encoder = FeatureTransformerEncoder(
            d_model=256, 
            nhead=8, 
            num_encoder_layers=6
        )
        
    def forward(self, images):
        """
        Args:
            images: [B, 3, H, W] RGB images
            
        Returns:
            enhanced_features: [B, 256, H', W'] Enhanced features for query initialization
        """
        # Extract multi-scale features using ResNet50
        resnet_features = self.resnet_backbone(images)  # [c2, c3, c4, c5]
        
        # Apply REM to highest level features
        rem_features = self.rem(resnet_features[-1])  # c5 -> enhanced features
        
        # Apply transformer encoder for global context
        enhanced_features = self.transformer_encoder(resnet_features, rem_features)
        
        return enhanced_features
