import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------#
#   Conv2d + Batch Normalization + LeakyReLU / Mish
# ---------------------------------------------------#
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, use_mish=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Mish() if use_mish else nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# ---------------------------------------------------#
#   Darknet53 Residual Block
# ---------------------------------------------------#
class DarknetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 1)
        self.conv2 = ConvBlock(out_channels, in_channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

# ---------------------------------------------------#
#   Modified/Standard Darknet53 Backbone
# ---------------------------------------------------#
class DarkNet53(nn.Module):
    def __init__(self, layers=[1, 2, 8, 8, 4]):
        super().__init__()
        self.in_channels = 32
        self.conv1 = ConvBlock(3, 32, 3, padding=1)
        
        # Five downsampling stages
        self.stage1 = self._make_stage(64, layers[0])
        self.stage2 = self._make_stage(128, layers[1])
        self.stage3 = self._make_stage(256, layers[2])
        self.stage4 = self._make_stage(512, layers[3])
        self.stage5 = self._make_stage(1024, layers[4])

    def _make_stage(self, out_channels, num_blocks):
        layers = []
        # Downsample with stride 2 conv
        layers.append(ConvBlock(self.in_channels, out_channels, 3, stride=2, padding=1))
        self.in_channels = out_channels
        for _ in range(num_blocks):
            layers.append(DarknetBasicBlock(self.in_channels, out_channels // 2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        c1 = self.stage1(x)  # stride 2
        c2 = self.stage2(c1) # stride 4
        c3 = self.stage3(c2) # stride 8
        c4 = self.stage4(c3) # stride 16
        c5 = self.stage5(c4) # stride 32
        return c2, c3, c4, c5  # Return multi-scale features

# ---------------------------------------------------#
#   ASPP-S Neck Module (Atrous Spatial Pyramid Pooling)
# ---------------------------------------------------#
class ASPPS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aspp1 = ConvBlock(in_channels, out_channels, 1)
        self.aspp2 = ConvBlock(in_channels, out_channels, 3, padding=6, dilation=6)
        self.aspp3 = ConvBlock(in_channels, out_channels, 3, padding=12, dilation=12)
        self.aspp4 = ConvBlock(in_channels, out_channels, 3, padding=18, dilation=18)
        self.project = ConvBlock(out_channels * 4, out_channels, 1)

    def forward(self, x):
        y1 = self.aspp1(x)
        y2 = self.aspp2(x)
        y3 = self.aspp3(x)
        y4 = self.aspp4(x)
        return self.project(torch.cat([y1, y2, y3, y4], dim=1))

# ---------------------------------------------------#
#   U-Shape Attention Module
# ---------------------------------------------------#
class SharedMLP(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

    def forward(self, x):
        return self.fc(x)

class UShapeAttention(nn.Module):
    def __init__(self, c1_channels, c2_channels, reduction=16):
        super().__init__()
        self.mlp1 = SharedMLP(c1_channels, reduction)
        self.mlp2 = SharedMLP(c2_channels, reduction)
        
        # Spatial attention convolutions (7x7 convs)
        self.spatial_conv1 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.spatial_conv2 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, F1, F2):
        # 1. Channel Attention for F1 (shallower)
        b, c1, _, _ = F1.size()
        avg_out1 = self.mlp1(F1.mean(dim=[2, 3])).view(b, c1, 1, 1)
        max_out1 = self.mlp1(F1.amax(dim=[2, 3])).view(b, c1, 1, 1)
        C1 = self.sigmoid(avg_out1 + max_out1)
        F1_prime = F1 * C1

        # 2. Channel Attention for F2 (deeper)
        b, c2, _, _ = F2.size()
        avg_out2 = self.mlp2(F2.mean(dim=[2, 3])).view(b, c2, 1, 1)
        max_out2 = self.mlp2(F2.amax(dim=[2, 3])).view(b, c2, 1, 1)
        C2 = self.sigmoid(avg_out2 + max_out2)
        F2_prime = self.upsample(F2) * C2

        # 3. Spatial Attention for F1_prime
        avg_spatial1 = torch.mean(F1_prime, dim=1, keepdim=True)
        max_spatial1, _ = torch.max(F1_prime, dim=1, keepdim=True)
        S1 = self.sigmoid(self.spatial_conv1(torch.cat([avg_spatial1, max_spatial1], dim=1)))
        F1_dprime = F1_prime * S1

        # 4. Spatial Attention for F2_prime
        avg_spatial2 = torch.mean(F2_prime, dim=1, keepdim=True)
        max_spatial2, _ = torch.max(F2_prime, dim=1, keepdim=True)
        S2 = self.sigmoid(self.spatial_conv2(torch.cat([avg_spatial2, max_spatial2], dim=1)))
        F2_dprime = F2_prime * S2

        # 5. Concatenation along channel dimension
        return torch.cat([F1_dprime, F2_dprime], dim=1)

# ---------------------------------------------------#
#   YOLO Head (Predicts bboxes & classes)
# ---------------------------------------------------#
class YoloHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels * 2, 3, padding=1),
            nn.Conv2d(in_channels * 2, num_anchors * (num_classes + 5), 1)
        )

    def forward(self, x):
        return self.conv(x)

# ---------------------------------------------------#
#   Main Model Body: Supports Baseline & Custom MSA-YOLO
# ---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes, custom=False):
        super().__init__()
        self.custom = custom
        
        # 1. Define Backbone
        if self.custom:
            # Paper's modified Darknet-53 repetitions [1, 4, 6, 6, 4]
            self.backbone = DarkNet53(layers=[1, 4, 6, 6, 4])
            
            # Neck: ASPP-S
            self.neck = ASPPS(1024, 512)
            
            # U-Shape Attentions for 4 scales
            # Connect c2-c3, c3-c4, c4-c5
            # Note: F2 input to each attention is the ADJUSTED output from the layer above
            self.attention_c4_c5 = UShapeAttention(512, 512)   # c4=512, c5_neck=512 → outputs 1024
            self.attention_c3_c4 = UShapeAttention(256, 256)   # c3=256, c4_out(adjusted)=256 → outputs 512
            self.attention_c2_c3 = UShapeAttention(128, 128)   # c2=128, c3_out(adjusted)=128 → outputs 256

            # Convolutions to adjust channels after attention blocks
            self.adjust_c5 = ConvBlock(512, 512, 1)
            self.adjust_c4 = ConvBlock(1024, 256, 1)
            self.adjust_c3 = ConvBlock(512, 128, 1)
            self.adjust_c2 = ConvBlock(256, 64, 1)

            # Four heads for four scales (152x152, 76x76, 38x38, 19x19)
            self.head1 = YoloHead(512, num_anchors, num_classes) # Scale 19x19 (from C5)
            self.head2 = YoloHead(256, num_anchors, num_classes) # Scale 38x38
            self.head3 = YoloHead(128, num_anchors, num_classes) # Scale 76x76
            self.head4 = YoloHead(64, num_anchors, num_classes)  # Scale 152x152 (shallower)
            
        else:
            # Standard YOLOv4 baseline
            self.backbone = DarkNet53(layers=[1, 2, 8, 8, 4])
            
            # Simple neck ConvBlocks (standard YOLOv3/v4 style)
            self.conv_c5 = ConvBlock(1024, 512, 1)
            self.conv_c4 = ConvBlock(512, 256, 1)
            self.conv_c3 = ConvBlock(256, 128, 1)
            
            # Three heads (scale 19x19, 38x38, 76x76)
            self.head1 = YoloHead(512, num_anchors, num_classes)
            self.head2 = YoloHead(256, num_anchors, num_classes)
            self.head3 = YoloHead(128, num_anchors, num_classes)

    def forward(self, x):
        # Backbone output features
        c2, c3, c4, c5 = self.backbone(x)
        
        if self.custom:
            # Custom MSA-YOLO architecture forward pass
            c5_neck = self.neck(c5)
            
            # Feature propagation via U-Shape Attention blocks
            f_c4_c5 = self.attention_c4_c5(c4, c5_neck)
            c4_out = self.adjust_c4(f_c4_c5)
            
            f_c3_c4 = self.attention_c3_c4(c3, c4_out)
            c3_out = self.adjust_c3(f_c3_c4)
            
            f_c2_c3 = self.attention_c2_c3(c2, c3_out)
            c2_out = self.adjust_c2(f_c2_c3)
            
            c5_out = self.adjust_c5(c5_neck)

            # Predict at 4 scales
            out1 = self.head1(c5_out)
            out2 = self.head2(c4_out)
            out3 = self.head3(c3_out)
            out4 = self.head4(c2_out)
            return [out1, out2, out3, out4]
            
        else:
            # Standard YOLOv4 forward pass
            c5_out = self.conv_c5(c5)
            c4_out = self.conv_c4(c4)
            c3_out = self.conv_c3(c3)
            
            out1 = self.head1(c5_out)
            out2 = self.head2(c4_out)
            out3 = self.head3(c3_out)
            return [out1, out2, out3]
