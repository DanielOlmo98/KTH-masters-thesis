import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import CenterCrop
from pytorch_wavelets import DWTForward, DWTInverse
from unet_model import ConvBlock


class WaveletUnet(nn.Module):
    def __init__(self, input_ch=1, output_ch=2, top_feature_ch=32, levels=4, **kwargs):
        super(WaveletUnet, self).__init__()
        self.out_ch = output_ch
        self.top_features = top_feature_ch
        self.levels = levels

        self.contracting_path = WaveletContractingPath(input_ch, self.top_features, self.levels)
        self.expanding_path = WaveletExpandingPath(self.top_features, self.levels)

        self.end = nn.Conv2d(top_feature_ch, self.out_ch, kernel_size=(1, 1), padding='same')

    def __str__(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f'New wavelet Unet:\n    ' \
               f'Levels: {self.levels}\n    ' \
               f'Top features: {self.top_features} \n' \
               f'Trainable parameters: {total_params}' \
               f'Output channels: {self.out_ch}'

    def forward(self, x):
        x, features = self.contracting_path(x)
        output = self.expanding_path(x, features[::-1])  # reverse
        output = self.end(output)

        return output

    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class WaveletContractingPath(nn.Module):
    def __init__(self, input_ch, top_features, levels):
        super(WaveletContractingPath, self).__init__()
        self.contr_path_blocks = self.get_contracting_path(input_ch, top_features, levels)
        self.pooling_layer = self.DwtDownsample()

    class DwtDownsample(nn.Module):
        def __init__(self):
            super().__init__()
            self.dwt = DWTForward(J=1, wave='db1').cuda()

        def forward(self, x):
            Yl, Yh = self.dwt(x)
            s_bands = torch.unbind(Yh[0], dim=2)
            x = torch.cat((Yl, s_bands[0], s_bands[1], s_bands[2]), dim=1)
            return x

    def get_contracting_path(self, input_ch, top_features, levels):
        contracting_path_blocks = nn.ModuleList()
        contracting_path_blocks.append(
            ConvBlock(input_ch, top_features)
        )
        print((input_ch, top_features))
        for level in range(levels - 2):
            top_features *= 2
            contracting_path_blocks.append(
                ConvBlock(top_features * 2, top_features)
            )
            print((top_features * 2, top_features))
        return contracting_path_blocks

    def forward(self, x):
        features = []
        for block in self.contr_path_blocks:
            x = block(x)
            features.append(x)
            print(x.shape)
            x = self.pooling_layer(x)
        return x, features


class WaveletExpandingPath(nn.Module):

    def __init__(self, input_ch, levels):
        super(WaveletExpandingPath, self).__init__()
        self.exp_path_upconv = self.IDwtUpsample()
        self.exp_path_blocks = self.get_expanding_path(input_ch, levels)

    class IDwtUpsample(nn.Module):
        def __init__(self):
            super().__init__()
            self.idwt = DWTInverse(wave='db1').cuda()

        def forward(self, x):
            s_bands = torch.chunk(x, 4, dim=1)
            Yh = [torch.stack((s_bands[1], s_bands[2], s_bands[3]), dim=2)]
            x = self.idwt((s_bands[0], Yh))
            return x

    def get_expanding_path(self, top_features, levels):
        exp_path_blocks = nn.ModuleList()
        for level in range(levels - 1):
            exp_path_blocks.append(
                ConvBlock(top_features * 2, top_features)
            )
            print((top_features * 2, top_features))
            top_features = top_features * 2
        exp_path_blocks.append(
            ConvBlock(top_features * 2, top_features * 2)
        )
        print((top_features * 2, top_features * 2))
        return exp_path_blocks[::-1]

    def forward(self, x, features):
        x = self.exp_path_blocks[0](x)
        for i in range(len(self.exp_path_blocks[1:])):
            print(i)
            print(x.shape)
            x = self.exp_path_upconv(x)

            print(x.shape)
            print(features[i].shape)
            feature_c = self._crop(x, features[i])
            x = torch.cat([x, feature_c], dim=1)
            x = self.exp_path_blocks[i + 1](x)
        return x

    def _crop(self, x, features):
        _, _, height, width = x.shape
        return CenterCrop([height, width])(features)
