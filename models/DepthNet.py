import torch
import torch.nn as nn
import torch.nn.functional as F
from .convnext import convnext_small


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

class DepthDecoder(nn.Module):
    def __init__(self, num_output_channels=1):
        super(DepthDecoder, self).__init__()
        self.alpha = 10
        self.beta = 0.01
        self.num_output_channels = num_output_channels

        # decoder
        self.upconv1_0 = ConvBlock(768, 512)
        self.upconv1_1 = ConvBlock(384+512, 512)
        self.upconv2_0 = ConvBlock(512, 256)
        self.upconv2_1 = ConvBlock(192 + 256, 256)
        self.upconv3_0 = ConvBlock(256, 128)
        self.upconv3_1 = ConvBlock(96 + 128, 96)

        self.upconv4 = ConvBlock(96, 128)
        self.upconv5 = ConvBlock(128, 256)
        self.out = Conv3x3(256, self.num_output_channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):

        x = input_features[-1]
        x = self.upconv1_0(x)
        x = [upsample(x), input_features[2]]
        x = torch.cat(x,1)
        x = self.upconv1_1(x)

        x = self.upconv2_0(x)
        x = [upsample(x), input_features[1]]
        x = torch.cat(x, 1)
        x = self.upconv2_1(x)

        x = self.upconv3_0(x)
        x = [upsample(x), input_features[0]]
        x = torch.cat(x, 1)
        x = self.upconv3_1(x)

        x = self.upconv4(upsample(x))
        x = self.upconv5(upsample(x))

        output = self.alpha * self.sigmoid(self.out(x)) + self.beta

        return output

class DepthNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DepthNet, self).__init__()
        self.encoder = convnext_small(num_input_images = 1, pretrained=True)
        self.decoder = DepthDecoder()

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)

        return output

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    model = DepthNet().cuda()
    model.train()

    B = 4

    tgt_img = torch.randn(B, 3, 256, 832).cuda()
    # ref_imgs = [torch.randn(B, 3, 256, 832).cuda() for i in range(2)]

    tgt_depth = model(tgt_img)

    print(tgt_depth.size())
