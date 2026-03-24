import torch
import torch.nn as nn

class ConvBlockVNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        layers = []
        for i in range(num_convs):
            if i == 0:
                layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2))
            else:
                layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=5, padding=2))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.PReLU(out_channels))

        self.conv_block = nn.Sequential(*layers)

        if in_channels != out_channels:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv_block(x)
        return out + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        return self.prelu(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        return self.prelu(self.up(x))


class VNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_channels=16):
        super().__init__()
        self.enc1 = ConvBlockVNet(in_channels, base_channels, num_convs=1)
        self.down1 = DownBlock(base_channels, base_channels * 2)

        self.enc2 = ConvBlockVNet(base_channels * 2, base_channels * 2, num_convs=2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)

        self.enc3 = ConvBlockVNet(base_channels * 4, base_channels * 4, num_convs=3)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)

        self.enc4 = ConvBlockVNet(base_channels * 8, base_channels * 8, num_convs=3)
        self.down4 = DownBlock(base_channels * 8, base_channels * 16)

        self.bottleneck = ConvBlockVNet(base_channels * 16, base_channels * 16, num_convs=3)

        self.up4 = UpBlock(base_channels * 16, base_channels * 8)
        self.dec4 = ConvBlockVNet(base_channels * 16, base_channels * 8, num_convs=3)

        self.up3 = UpBlock(base_channels * 8, base_channels * 4)
        self.dec3 = ConvBlockVNet(base_channels * 8, base_channels * 4, num_convs=3)

        self.up2 = UpBlock(base_channels * 4, base_channels * 2)
        self.dec2 = ConvBlockVNet(base_channels * 4, base_channels * 2, num_convs=2)

        self.up1 = UpBlock(base_channels * 2, base_channels)
        self.dec1 = ConvBlockVNet(base_channels * 2, base_channels, num_convs=1)

        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        d1 = self.down1(s1)

        s2 = self.enc2(d1)
        d2 = self.down2(s2)

        s3 = self.enc3(d2)
        d3 = self.down3(s3)

        s4 = self.enc4(d3)
        d4 = self.down4(s4)

        b = self.bottleneck(d4)

        u4 = self.up4(b)
        u4 = torch.cat([u4, s4], dim=1)
        dec4 = self.dec4(u4)

        u3 = self.up3(dec4)
        u3 = torch.cat([u3, s3], dim=1)
        dec3 = self.dec3(u3)

        u2 = self.up2(dec3)
        u2 = torch.cat([u2, s2], dim=1)
        dec2 = self.dec2(u2)

        u1 = self.up1(dec2)
        u1 = torch.cat([u1, s1], dim=1)
        dec1 = self.dec1(u1)

        return self.out_conv(dec1)