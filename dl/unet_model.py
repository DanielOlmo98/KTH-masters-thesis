import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import CenterCrop
from pytorch_wavelets import DWTForward, DWTInverse


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
            print((input_ch, n_channels))
            input_ch = n_channels
        return contracting_path_blocks

    def forward(self, x):
        features = []
        for block in self.contr_path_blocks:
            x = block(x)
            features.append(x)
            x = self.pooling_layer(x)
        return features


class ExpandingPath(nn.Module):

    def __init__(self, channels, upsample_layer):
        super(ExpandingPath, self).__init__()
        self.channels = reversed(channels)
        self.exp_path_upconv, self.exp_path_blocks = self.get_expanding_path(upsample_layer)

    def get_expanding_path(self, upsample_layer):
        exp_path_upconv = nn.ModuleList()
        exp_path_blocks = nn.ModuleList()
        print('expand')
        for n_channels in self.channels:
            exp_path_upconv.append(  # use lambda???
                upsample_layer(n_channels)
            )
            exp_path_blocks.append(
                ConvBlock(n_channels * 2, n_channels)
            )
            print((n_channels * 2, n_channels))
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


class Unet(nn.Module):

    def __init__(self, input_ch=1, output_ch=2, top_feature_ch=32, levels=4):
        super(Unet, self).__init__()
        self.out_ch = output_ch
        self.channels = torch.logspace(np.log2(top_feature_ch), np.log2(top_feature_ch) + levels - 1, levels, 2,
                                       dtype=torch.int)
        self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample_layer = lambda n_ch: nn.ConvTranspose2d(n_ch * 2, n_ch, kernel_size=(2, 2), stride=(2, 2))

        self.contracting_path = ContractingPath(input_ch, self.channels, self.pooling_layer)
        self.expanding_path = ExpandingPath(self.channels[:-1], self.upsample_layer)

        self.end = nn.Conv2d(self.channels[0], self.out_ch, kernel_size=(1, 1), padding='same')

    def __str__(self):
        return f'Unet:\n    ' \
               f'Levels: {self.channels.size()[0]}\n    ' \
               f'Features per level: {self.channels}\n    ' \
               f'Pooling layer: {self.pooling_layer}\n    ' \
               f'Output channels: {self.out_ch}'

    def forward(self, x):
        features = self.contracting_path(x)
        output = self.expanding_path(features[::-1][0], features[::-1][1:])  # reverse
        output = self.end(output)

        return output

    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class WaveletUnet(nn.Module):
    def __init__(self, input_ch=1, output_ch=2, top_feature_ch=32, levels=4):
        super(WaveletUnet, self).__init__()
        self.out_ch = output_ch
        self.top_features = top_feature_ch
        self.levels = levels

        self.contracting_path = WaveletContractingPath(input_ch, self.top_features, self.levels)
        self.expanding_path = WaveletExpandingPath(self.top_features, self.levels)

        self.end = nn.Conv2d(top_feature_ch, self.out_ch, kernel_size=(1, 1), padding='same')

    def __str__(self):
        return f'Unet:\n    ' \
               f'Levels: {self.levels}\n    ' \
               f'Top features: {self.top_features} \n' \
               f'Output channels: {self.out_ch}'

    def forward(self, x):
        features = self.contracting_path(x)
        output = self.expanding_path(features[::-1][0], features[::-1][1:])  # reverse
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
        for level in range(levels-1):
            top_features *= 4
            contracting_path_blocks.append(
                ConvBlock(top_features, top_features)
            )
            print((top_features, top_features))
        return contracting_path_blocks

    def forward(self, x):
        features = []
        for block in self.contr_path_blocks:
            x = block(x)
            features.append(x)
            x = self.pooling_layer(x)
        return features


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
        for level in range(levels-1):
            exp_path_blocks.append(
                ConvBlock(top_features * 2, top_features)
            )
            print((top_features * 2, top_features))
            top_features = top_features * 4
        return exp_path_blocks[::-1]

    def forward(self, x, features):
        for i in range(len(self.exp_path_blocks)):
            x = self.exp_path_upconv(x)
            feature_c = self._crop(x, features[i])
            x = torch.cat([x, feature_c], dim=1)
            x = self.exp_path_blocks[i](x)
        return x

    def _crop(self, x, features):
        _, _, height, width = x.shape
        return CenterCrop([height, width])(features)


import unittest
from torchviz import make_dot


class UnetTest(unittest.TestCase):
    def test(self):
        unet_settings = {
            'levels': 5,
            'top_feature_ch': 16,
            'output_ch': 1
        }
        model = Unet(**unet_settings)
        x = torch.randn((3, 1, 256, 256))
        preds = model(x)
        print(model)
        print(x.shape)
        print(preds.shape)

        make_dot(preds, params=dict(model.named_parameters())).render("net5", format="png")

        assert preds.shape == x.shape


if __name__ == '__main__':
    unittest.main()
    # unet_settings = {
    #     'levels': 3,
    #     'top_feature_ch': 16,
    #     'output_ch': 1
    # }
    # model = Unet(**unet_settings)
    # x = torch.randn((4, 1, 256, 256)).type(torch.float32)
    # y = model(x)
    # input_names = ['img']
    # output_names = ['seg']
    # torch.onnx.export(model, x, 'unet.onnx', input_names=input_names, output_names=output_names)
