import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch, types
from torch import nn
import torch.nn.functional as F
from ..utils.utils import *
from .SD_v1_5.backprop import RevModule, VanillaBackProp, RevBackProp
from .SD_v1_5.forward import MyUNet2DConditionModel_SD_v1_5_forward


"""
A set of functions for integrating ChannelAttn into the UNet.

These modules are designed to be inserted serially into the UNet backbone 
and ResNet blocks, with concrete implementation details provided.
Since the pretrained model is encapsulated within the `diffusers` library, 
modifications must be performed either by registering new modules 
or via dynamic replacement.

Note that these modules are experimental, as they may significantly 
interfere with the pretrained weights. You may adapt the implementation
details below or follow a similar way to integrate your own custom modules.

Overview:

class CALayer
 - Implementation of a channel attention layer.

Function List:
- unet_add_channel_attn: Adds channel attention layers to the UNet.
- unet_set_resnet_attn: Injects channel attention into ResNet blocks within the UNet.
- ResBlock_forward_with_attn: Replaces the forward method of ResNet blocks.
- UNet_forward_with_attn: Replaces the forward method of the UNet.
"""


class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def unet_add_channel_attn(self, channels):
    self.unet.register_module("UNet_CA", CALayer(channels))
    self.unet.register_module("uca", nn.ModuleList([RevModule(self.unet.UNet_CA)]))
    self.unet.register_parameter("uca_input_help_scale_factor", nn.Parameter(torch.ones(1,)))
    self.unet.register_parameter("uca_merge_scale_factors", nn.Parameter(torch.zeros(1,)))


def unet_set_resnet_attn(self, enable_dense_net=False):
    """
    We provide two implementations: ResNet and DenseNet.
    """

    from diffusers.models.resnet import ResnetBlock2D
    def ResnetBlock2D_add_ca(module):
        if isinstance(module, ResnetBlock2D):
            module.ca = CALayer(module.out_channels)
            if enable_dense_net:
                module.dense_conv_11 = nn.Conv2d(
                    in_channels  = module.out_channels * 2,
                    out_channels = module.out_channels,
                    kernel_size  = 1,
                    padding      = 0,
                    bias         = False
                )
    self.unet.apply(ResnetBlock2D_add_ca)


def UNet_forward_with_attn(self, x):
    x = torch.cat([x, self.uca_input_help_scale_factor * x], dim=1)
    x = RevBackProp.apply(x, self.uca)
    x_split = x.chunk(2, dim=1)
    x_merge = x_split[0] + self.uca_merge_scale_factors * x_split[1]
    x = F.pixel_unshuffle(x_merge, 2)

    return MyUNet2DConditionModel_SD_v1_5_forward(self, x)


def ResBlock_forward_with_attn(self, x_in):
    if hasattr(self, "skip"):
        x_in = torch.cat([x_in, self.skip], dim=1)
    x = self.norm1(x_in)
    x = self.nonlinearity(x)
    x = self.conv1(x)
    x = self.norm2(x)
    x = self.nonlinearity(x)
    x = self.conv2(x)

    # Apply ChannelAttn in ResBlock
    x = self.ca(x)
    
    if hasattr(self, "dense_conv_11"):
        if self.in_channels == self.out_channels:
            x = torch.cat([x, x_in], dim=1)
        else:
            x = torch.cat([x, self.conv_shortcut(x_in)], dim=1)
        x = self.dense_conv_11(x)    
    else:
        if self.in_channels == self.out_channels:
            x = x + x_in
        else:
            x = x + self.conv_shortcut(x_in)
        
    return x
