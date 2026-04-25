import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3Encoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        # multi-scale features
        # num_layer: (0, 1, 3, 8, 12)
        # num_chan: (16, 16, 24, 48, 576)
        features = backbone.features
        self.stage1 = nn.Sequential(*features[0: 1])
        self.stage2 = nn.Sequential(*features[1: 2])
        self.stage3 = nn.Sequential(*features[2: 4])
        self.stage4 = nn.Sequential(*features[4: 9])
        self.stage5 = nn.Sequential(*features[9:])

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return x1, x2, x3, x4, x5


class LRASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.global_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.Hardsigmoid(),)
        self.local_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        global_feat = self.global_attn(x)  # global context attention weight
        local_feat = self.local_conv(x)  # local feature operation
        return local_feat * global_feat


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, low_level_channels, aspp_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.low_level_reduce = nn.Conv2d(low_level_channels, 64, kernel_size=1)
        self.aspp = LRASPP(aspp_channels, 128)
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),)
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),)
        self.output = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, low_level_feat, aspp_feat):
        low_level_feat = self.low_level_reduce(low_level_feat)
        aspp_feat = F.interpolate(
            self.aspp(aspp_feat), size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([aspp_feat, low_level_feat], dim=1)
        x = self.decoder_conv2(self.decoder_conv1(x))
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.output(x) # [B, num_classes, H, W]
        return x if self.num_classes != 1 else F.sigmoid(x)


class MobileNetV3_DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.encoder = MobileNetV3Encoder(backbone)
        self.decoder = DeepLabV3PlusDecoder(
            low_level_channels=16, aspp_channels=48, num_classes=num_classes)

    def forward(self, x):
        c1, c2, c3, c4, c5 = self.encoder(x)
        out = self.decoder(c2, c4)
        return F.sigmoid(out) if self.num_classes == 1 else out


class SegmentationHead(nn.Module):
    def __init__(self, num_classes, s_chan, h_chan, img_size, hid_chan=19):
        super().__init__()
        self.num_classes = num_classes
        self.conv_s_1 = nn.Sequential(
            nn.Conv2d(s_chan, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),)
        self.conv_s_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(s_chan, 128, kernel_size=1, stride=1, padding=0), nn.Sigmoid(), 
            nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=True),)
        self.conv_s = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(128, hid_chan, kernel_size=1, stride=1, padding=0),)
        self.conv_h = nn.Conv2d(h_chan, hid_chan, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            nn.Conv2d(hid_chan, num_classes, kernel_size=1, stride=1, padding=0),)

    def forward(self, x1, x2):
        x_attn = self.conv_s_2(x2)
        x2, x1 = self.conv_s_1(x2), self.conv_h(x1)
        out = self.output(self.conv_s(x2 * x_attn) + x1)
        return F.sigmoid(out) if self.num_classes == 1 else out


class MobileNetV3Seg(nn.Module):
    def __init__(self, num_classes, hid_chan=19, img_size=256):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.encoder = MobileNetV3Encoder(backbone)
        self.decoder = SegmentationHead(
            num_classes, s_chan=576, h_chan=24, img_size=256//32, hid_chan=19)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        out = self.decoder(x3, x5)
        return out


if __name__ == "__main__":
    x = torch.randn(8, 3, 256, 256).cuda()
    model1 = MobileNetV3_DeepLabV3Plus(num_classes=1).cuda()
    model2 = MobileNetV3Seg(num_classes=1).cuda()
    print(model1(x).shape, model2(x).shape)
