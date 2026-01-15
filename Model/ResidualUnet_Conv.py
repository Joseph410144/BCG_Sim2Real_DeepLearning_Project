import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    (2x) Conv 3x3 + Norm + LeakyReLU
    """
    def __init__(self, in_channels, out_channels, norm="bn"):
        super().__init__()

        def NormLayer(c):
            if norm == "bn":
                return nn.BatchNorm2d(c)
            elif norm == "gn":
                # 比 BN 更穩（小 batch 推薦）
                return nn.GroupNorm(num_groups=min(8, c), num_channels=c)
            elif norm == "in":
                return nn.InstanceNorm2d(c, affine=True)
            else:
                return nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            NormLayer(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            NormLayer(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualUNetSCG(nn.Module):
    """
    Residual U-Net denoiser for scalogram-like 2D inputs.
    Input : (B, C=3, Freq, Time)
    Output: (B, C=3, Freq, Time)  # denoised = x - noise_pred
    """
    def __init__(self, in_channels=3, out_channels=3, base_ch=48, norm="bn"):
        super().__init__()

        # ---------------- Encoder ----------------
        self.enc1 = ConvBlock(in_channels, base_ch, norm=norm)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))

        self.enc2 = ConvBlock(base_ch, base_ch, norm=norm)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))

        self.enc3 = ConvBlock(base_ch, base_ch, norm=norm)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1))

        self.enc4 = ConvBlock(base_ch, base_ch, norm=norm)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1))

        self.enc5 = ConvBlock(base_ch, base_ch, norm=norm)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 1))

        # ---------------- Bottleneck ----------------
        # 先用 ConvBlock baseline（比 seq_len=1 的 ConvLSTM 更合理）
        self.bottleneck = ConvBlock(base_ch, base_ch, norm=norm)

        # ---------------- Decoder ----------------
        self.up7 = nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True)
        self.dec7 = ConvBlock(base_ch + base_ch, 96, norm=norm)    # 48+48 -> 96

        self.up8 = nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True)
        self.dec8 = ConvBlock(96 + base_ch, 96, norm=norm)         # 96+48 -> 96

        self.up9 = nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True)
        self.dec9 = ConvBlock(96 + base_ch, 96, norm=norm)

        self.up10 = nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True)
        self.dec10 = ConvBlock(96 + base_ch, 96, norm=norm)

        self.up11 = nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True)
        self.dec11 = ConvBlock(96 + base_ch, 24, norm=norm)

        # predict noise
        self.final_conv = nn.Conv2d(24, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)      # (B,48,F,T)
        p1 = self.pool1(e1)    # (B,48,F/2,T)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        e5 = self.enc5(p4)
        p5 = self.pool5(e5)

        # Bottleneck
        b = self.bottleneck(p5)

        # Decoder
        u7 = self.up7(b)
        if u7.shape[2:] != e5.shape[2:]:
            u7 = F.interpolate(u7, size=e5.shape[2:])
        d7 = self.dec7(torch.cat([u7, e5], dim=1))

        u8 = self.up8(d7)
        if u8.shape[2:] != e4.shape[2:]:
            u8 = F.interpolate(u8, size=e4.shape[2:])
        d8 = self.dec8(torch.cat([u8, e4], dim=1))

        u9 = self.up9(d8)
        if u9.shape[2:] != e3.shape[2:]:
            u9 = F.interpolate(u9, size=e3.shape[2:])
        d9 = self.dec9(torch.cat([u9, e3], dim=1))

        u10 = self.up10(d9)
        if u10.shape[2:] != e2.shape[2:]:
            u10 = F.interpolate(u10, size=e2.shape[2:])
        d10 = self.dec10(torch.cat([u10, e2], dim=1))

        u11 = self.up11(d10)
        if u11.shape[2:] != e1.shape[2:]:
            u11 = F.interpolate(u11, size=e1.shape[2:])
        d11 = self.dec11(torch.cat([u11, e1], dim=1))

        noise_pred = self.final_conv(d11)
        denoised = x - noise_pred
        return denoised


# ---------------- Quick shape test ----------------
if __name__ == "__main__":
    x = torch.randn(4, 3, 512, 256)
    model = ResidualUNetSCG(in_channels=3, out_channels=3, norm="bn")
    y = model(x)
    print("Input:", x.shape, "Output:", y.shape)
    assert x.shape == y.shape
    print("Shape check passed!")
