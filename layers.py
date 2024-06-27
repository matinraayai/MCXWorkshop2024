import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    """
    The original res block as described in the resnet paper.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, do_3d=False,
                 kernel_size=3, stride=1, dilation=1,
                 padding=1,
                 padding_mode='replicate',
                 projection=True,
                 activation_fn=F.relu):
        super(ResidualBlock, self).__init__()
        conv_layer = nn.Conv3d if do_3d else nn.Conv2d
        norm_layer = nn.BatchNorm3d if do_3d else nn.BatchNorm2d
        self.conv1 = conv_layer(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, padding_mode=padding_mode,
                                bias=False,
                                dilation=dilation)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = conv_layer(out_channels, out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, padding_mode=padding_mode,
                                bias=False,
                                dilation=dilation)
        self.bn2 = norm_layer(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if projection:
                self.shortcut = nn.Sequential(conv_layer(in_channels, out_channels, kernel_size=1, padding=0),
                                              norm_layer(out_channels))
            else:
                self.shortcut = lambda x: x
        self.activation_fn = activation_fn

    def forward(self, x):
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation_fn(out)
        return out


class DnCNN(nn.Module):
    def __init__(self, do_3d=False, kernel_size=3, padding=1, padding_mode='reflect', input_channels=1,
                 output_channels=1,
                 inter_kernel_channel=64, num_layers=17, activation_fn=F.relu):
        super(DnCNN, self).__init__()
        self.num_layers = num_layers
        self.activation_fn = activation_fn
        conv_layer = nn.Conv3d if do_3d else nn.Conv2d
        norm_layer = nn.BatchNorm3d if do_3d else nn.BatchNorm2d
        self.__setattr__("conv1", conv_layer(input_channels,
                                             inter_kernel_channel,
                                             kernel_size,
                                             padding_mode=padding_mode,
                                             padding=padding))
        for num_layer in range(2, num_layers):
            self.__setattr__(f"conv{num_layer}", conv_layer(inter_kernel_channel,
                                                            inter_kernel_channel,
                                                            kernel_size,
                                                            padding_mode=padding_mode,
                                                            padding=padding,
                                                            bias=False))
            self.__setattr__(f"norm{num_layer}", norm_layer(inter_kernel_channel))
        self.__setattr__(f"conv{num_layers}", conv_layer(inter_kernel_channel,
                                                         output_channels,
                                                         kernel_size,
                                                         padding_mode=padding_mode,
                                                         padding=padding))

    def forward(self, x):
        output = self.activation_fn(self.conv1(x))
        for num_layer in range(2, self.num_layers):
            output = self.__getattr__(f"conv{num_layer}")(output)
            output = self.__getattr__(f"norm{num_layer}")(output)
            output = self.activation_fn(output)
        output = self.__getattr__(f"conv{self.num_layers}")(output)

        return x - output
