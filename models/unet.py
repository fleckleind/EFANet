import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = Block(in_ch, out_ch)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.down(out1)
        return out1, out2


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, skip=True):
        super().__init__()
        self.skip = skip
        self.conv = Block(in_ch, out_ch)
        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, out1):
        out = self.upSample(x)
        if self.skip:
            out = torch.cat((out, out1), dim=1)
        out = self.conv(out)
        return out


class Unet(nn.Module):
    def __init__(self, img_channel, num_classes, skip=True):
        super().__init__()
        channel = [32, 64, 128, 256, 512]
        self.skip = skip
        self.img_channel = img_channel
        self.num_classes = num_classes
        # encoder layers
        self.e1 = Encoder(self.img_channel, channel[0])
        self.e2 = Encoder(channel[0], channel[1])
        self.e3 = Encoder(channel[1], channel[2])
        self.e4 = Encoder(channel[2], channel[3])
        # bottleneck layer
        self.bn = Block(channel[3], channel[4])
        if self.skip:
            # decoder layers with skip connection
            self.d4 = Decoder(channel[4] + channel[3], channel[3])
            self.d3 = Decoder(channel[3] + channel[2], channel[2])
            self.d2 = Decoder(channel[2] + channel[1], channel[1])
            self.d1 = Decoder(channel[1] + channel[0], channel[0])
        else:
            # decoder layers without skip connection
            self.d4 = Decoder(channel[4], channel[3], skip=self.skip)
            self.d3 = Decoder(channel[3], channel[2], skip=self.skip)
            self.d2 = Decoder(channel[2], channel[1], skip=self.skip)
            self.d1 = Decoder(channel[1], channel[0], skip=self.skip)
        # output layer
        self.output = nn.Sequential(
            nn.Conv2d(channel[0], self.num_classes, 1, 1, 0),
        )

    def forward(self, x):
        # contraction path
        out1, out = self.e1(x)
        out2, out = self.e2(out)
        out3, out = self.e3(out)
        out4, out = self.e4(out)
        # bottleneck layer
        out = self.bn(out)
        # expanding path
        out = self.d4(out, out4)
        out = self.d3(out, out3)
        out = self.d2(out, out2)
        out = self.d1(out, out1)
        # segmentation
        out = self.output(out)
        if self.num_classes == 1:
            out = torch.sigmoid(out)
        return out
