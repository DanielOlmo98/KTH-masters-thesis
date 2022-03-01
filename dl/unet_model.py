import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import CenterCrop


class ConvBlock(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_ch, output_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ExpandingPath(nn.Module):

    def __init__(self, channels):
        super(ExpandingPath, self).__init__()
        self.channels = reversed(channels)
        self.exp_path_upconv, self.exp_path_blocks = self.get_expanding_path()

    def get_expanding_path(self):
        exp_path_upconv = nn.ModuleList()
        exp_path_blocks = nn.ModuleList()
        for n_channels in self.channels:
            exp_path_upconv.append(
                nn.ConvTranspose2d(n_channels * 2, n_channels, kernel_size=(2, 2), stride=(2, 2))
            )
            exp_path_blocks.append(
                ConvBlock(n_channels * 2, n_channels)
            )
        return exp_path_upconv, exp_path_blocks

    def forward(self, x, features):
        for i in range(len(self.exp_path_upconv)):
            x = self.exp_path_upconv[i](x)
            feature_c = self._crop(x, features[i])
            x = torch.cat([x, feature_c], dim=1)
            x = self.exp_path_blocks[i](x)
        return x

    def _crop(self, x, features):
        _, _, height, width = x.shape
        return CenterCrop([height, width])(features)


class ContractingPath(nn.Module):
    def __init__(self, input_ch, channels, pooling_layer):
        super(ContractingPath, self).__init__()
        self.channels = channels
        self.contr_path_blocks = self.get_contracting_path(input_ch)
        self.pooling_layer = pooling_layer

    def get_contracting_path(self, input_ch):
        contracting_path_blocks = nn.ModuleList()
        for n_channels in self.channels:
            contracting_path_blocks.append(
                ConvBlock(input_ch, n_channels)
            )
            input_ch = n_channels
        return contracting_path_blocks

    def forward(self, x):
        features = []
        for block in self.contr_path_blocks:
            x = block(x)
            features.append(x)
            x = self.pooling_layer(x)
        return features


class Unet(nn.Module):

    def __init__(self, input_ch=1, output_ch=2, top_feature_ch=32, levels=4):
        super(Unet, self).__init__()
        self.channels = torch.logspace(np.log2(top_feature_ch), np.log2(top_feature_ch) + levels - 1, levels, 2,
                                       dtype=torch.int)

        self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        self.contracting_path = ContractingPath(input_ch, self.channels, self.pooling_layer)
        self.expanding_path = ExpandingPath(self.channels[:-1])

        self.end = nn.Conv2d(self.channels[0], output_ch, kernel_size=(1, 1), padding='same')

    def forward(self, x):
        features = self.contracting_path(x)
        output = self.expanding_path(features[::-1][0], features[::-1][1:])  # reverse
        output = self.end(output)

        return output


import unittest


class UnetTest(unittest.TestCase):
    def test(self):
        x = torch.randn((3, 1, 161, 161))
        model = Unet(input_ch=1, output_ch=1)
        preds = model(x)
        print(model)
        print(x.shape)
        print(preds.shape)
        assert preds.shape == x.shape


if __name__ == '__main__':
    unittest.main()
