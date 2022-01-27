import torch
import torch.nn as nn
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_ch, output_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ExpandingPath(nn.Module):

    def __init__(self, features):
        super(ExpandingPath, self).__init__()
        self.features = features

    def get_expanding_path(self):
        expanding_path_module = nn.ModuleList()
        for feature in reversed(self.features):
            expanding_path_module.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=(2, 2), stride=(2, 2))
            )
            expanding_path_module.append(
                ConvBlock(feature * 2, feature)
            )
        return expanding_path_module


class ContractingPath(nn.Module):
    def __init__(self, input_ch, features, pooling_layer):
        super(ContractingPath, self).__init__()
        self.features = features
        self.pooling_layer = pooling_layer
        self.path = self.get_contracting_path(input_ch)

    def get_contracting_path(self, input_ch):
        contracting_path_module = nn.ModuleList()
        for feature in self.features:
            contracting_path_module.append(
                ConvBlock(input_ch, feature)
            )
            input_ch = feature
        return contracting_path_module

    def forward(self, x):
        features = []
        for block in self.path:
            x = block(x)
            features.append(x)
            x = self.pooling_layer(x)
        return features


class Unet(nn.Module):

    def __init__(self, input_ch=1, output_ch=1, top_features=32, levels=4):
        super(Unet, self).__init__()
        self.features = torch.logspace(np.log2(top_features), np.log2(top_features) + levels - 1, levels, 2,
                                       dtype=torch.int)[1:]
        self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        self.contracting_path = self.get_contracting_path(input_ch)
        self.expanding_path = self.get_expanding_path()

        self.bottom = ConvBlock(self.features[-1], self.features[-1] * 2)
        self.end = nn.Conv2d(self.features[0], output_ch, kernel_size=(1, 1))


if __name__ == '__main__':
