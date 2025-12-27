import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch, types
from torch import nn
import torch.nn.functional as F
from ..utils.utils import *
from .SD_v1_5.backprop import RevModule, VanillaBackProp, RevBackProp
from .SD_v1_5.model import Injector, Step, Net as BaseNet


# configs
S = 28
C = 32
Q = 3


class My:
    @staticmethod
    def Injector():
        """
        Define Injector class for MSCD.
        """
        
        def __init__(self, nf, r, T):
            super(Injector, self).__init__()
            self.f2i = nn.ModuleList([nn.Sequential(
                nn.PixelShuffle(r),
                nn.Conv2d(nf//(r*r), S, 1),
            ) for _ in range(T)])
            
            self.ssg = nn.ModuleList([nn.Sequential(
                nn.Conv2d(S, C, kernel_size=3, stride=1, padding=1),
                *[nn.Sequential(
                    nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(C),
                    nn.ReLU(inplace=True)
                ) for _ in range(Q)],
                nn.Conv2d(C, S, kernel_size=3, stride=1, padding=1),
                nn.Sequential(
                    nn.Conv2d(S, S, kernel_size=3, stride=1, padding=1),
                    nn.Tanh()
                )
            ) for _ in range(T)])

            self.group = nn.ModuleList([
                nn.Conv2d(3*S, S, kernel_size=1, groups=S) for _ in range(T)
            ])
            
            self.i2f = nn.ModuleList([nn.Sequential(
                nn.Conv2d(S, nf//(r*r), 1),
                nn.PixelUnshuffle(r),
            ) for _ in range(T)])

        def forward(self, x_in):
            """
            Scaling the input and applying measurement physics {ATAf, ATy}
            to generate multiple solutions for the inverse problem.
            """

            f = self.f2i[t-1](x_in)
            s_0 = self.ssg[t-1][0](f)
            s_q = s_0
            
            for i in range(1, len(self.ssg[t-1]) - 2):
                s_q = self.ssg[t-1][i](s_q)
            
            s_q = s_q + s_0
            s_map = self.ssg[t-1][-2](s_q)
            step_map = self.ssg[t-1][-1](s_map) + 1
            
            b, s, h, w = f.shape
            ata_f = torch.zeros_like(f)
            at_y = torch.zeros_like(f)
            f_solutions = f.view(b, s, 1, h, w)
            
            for i in range(S):
                f_current = f_solutions[:, i]
                residual = A(f_current)
                ata_current = AT(residual)
                at_y_current = ATy

                ata_f[:, i:i+1] = ata_current
                at_y[:, i:i+1] = at_y_current
                
            ata_f = ata_f * step_map
            at_y = at_y * step_map
            
            x_rearr = []
            for i in range(S):
                x_rearr.append(f[:, i:i+1])
                x_rearr.append(ata_f[:, i:i+1])
                x_rearr.append(at_y[:, i:i+1])
            
            x = torch.cat(x_rearr, dim=1)
            x = self.group[t-1](x)
            return x_in + self.i2f[t-1](x)

        # Replace original Injector
        Injector.__init__ = __init__
        Injector.forward = forward

        return Injector

    @staticmethod
    def Step():
        def body(self, x):
            """
            DDNM Sampling Step for MSCD.
            Including 4 steps:
            
                0. Noise Estimation
                1. Denoising
                2. RND Correction for Multi-Solution
                3. Inverse DDIM Sampling
            """

            with torch.cuda.amp.autocast(enabled=use_amp, cache_enabled=False):
                global t
                t = self.t
                cur_alpha_bar = alpha_bar[t]
                prev_alpha_bar = alpha_bar[t-1]

                # 0. Noise Estimation
                e = F.pixel_shuffle(unet(F.pixel_unshuffle(x, 2)), 2)
                
                # 1. Denoising
                x = (x - (1 - cur_alpha_bar).pow(0.5) * e) / cur_alpha_bar.pow(0.5)

                # 2. RND for Multi-Solution
                # x = x - AT(A(x) - y)
                b, sc, h, w = x.shape
                c = sc // S
                x_solutions = x.view(b, S, c, h, w)
                x_corrected = torch.zeros_like(x)

                # Calculate correction for each solution
                for i in range(S):
                    x_current = x_solutions[:, i]
                    residual = A(x_current) - y
                    correction = AT(residual)
                    x_corrected[:, i:i+1] = x_current - correction
                    
                x = x_corrected

                # 3. Inverse DDIM Sampling
                return prev_alpha_bar.pow(0.5) * x + (1 - prev_alpha_bar).pow(0.5) * e

        # Apply
        Step.body = body
        return Step

    @staticmethod
    def Net():

        Injector = My.Injector()
        Step = My.Step()

        def __init__(self, T, unet):
            super(BaseNet, self).__init__()
            del unet.time_embedding, unet.mid_block
            unet.down_blocks = unet.down_blocks[:-2]
            unet.down_blocks[-1].downsamplers = None
            unet.up_blocks = unet.up_blocks[2:]

            self.body = nn.ModuleList([Step(T-i) for i in range(T)])
            self.input_help_scale_factors = nn.Parameter(torch.ones(S))
            self.merge_scale_factors = nn.Parameter(torch.zeros(S))
            self.alpha = nn.Parameter(torch.full((T,), 0.5))
            self.unet = unet
            self.fusion = nn.Conv2d(S, 1, kernel_size=1, bias=True)

            self.unet_add_down_rev_modules_and_injectors(T)
            self.unet_add_up_rev_modules_and_injectors(T)
            self.unet_remove_resnet_time_emb_proj()
            self.unet_remove_cross_attn()
            self.unet_set_inplace_to_true()
            self.unet_replace_forward_methods()

            # Use a pair of Conv2d instead of VAEs for encoding and decoding
            self.unet_set_conv_io()

        def unet_set_conv_io(self):
            ori_conv_in = self.unet.conv_in
            ori_conv_in_in_channels = ori_conv_in.in_channels
            ori_conv_in_out_channels = ori_conv_in.out_channels
            ori_conv_in_kernel_size = ori_conv_in.kernel_size
            ori_conv_in_stride = ori_conv_in.stride
            ori_conv_in_padding = ori_conv_in.padding
            ori_conv_in_bias = ori_conv_in.bias is not None

            multi_in_channels = ori_conv_in_in_channels * S
            multi_conv_in = nn.Conv2d(
                in_channels  = multi_in_channels,
                out_channels = ori_conv_in_out_channels,
                kernel_size  = ori_conv_in_kernel_size,
                stride       = ori_conv_in_stride,
                padding      = ori_conv_in_padding,
                bias         = ori_conv_in_bias
            )

            self.unet.conv_in = multi_conv_in
            self.unet.config.in_channels = multi_in_channels

            ori_conv_out = self.unet.conv_out
            ori_conv_out_in_channels = ori_conv_out.in_channels
            ori_conv_out_out_channels = ori_conv_out.out_channels
            ori_conv_out_kernel_size = ori_conv_out.kernel_size
            ori_conv_out_stride = ori_conv_out.stride
            ori_conv_out_padding = ori_conv_out.padding
            ori_conv_out_bias = ori_conv_out.bias is not None

            multi_out_channels = ori_conv_out_out_channels * S
            multi_conv_out = nn.Conv2d(
                in_channels  = ori_conv_out_in_channels,
                out_channels = multi_out_channels,
                kernel_size  = ori_conv_out_kernel_size,
                stride       = ori_conv_out_stride,
                padding      = ori_conv_out_padding,
                bias         = ori_conv_out_bias
            )

            self.unet.conv_out = multi_conv_out
            self.unet.config.out_channels = multi_out_channels

        def forward(self, y_, A_, AT_, use_amp_=True):
            global y, A, AT, unet, ATy, alpha_bar, use_amp
            y, A, AT, unet, use_amp = y_, A_, AT_, self.unet, use_amp_
            alpha_bar = torch.cat([torch.ones(1, device=y.device), self.alpha.cumprod(dim=0)])

            x = AT(y)
            ATy = x
            x = x.repeat(1, S, 1, 1)

            b, sc, h, w = x.shape
            c = sc // S
            x_reshaped = x.view(b, S, c, h, w)

            input_help_scales = self.input_help_scale_factors.view(1, S, 1, 1, 1)
            scaled_x = input_help_scales * x_reshaped
            x_original = x_reshaped.reshape(b, sc, h, w)
            x_scaled = scaled_x.reshape(b, sc, h, w)

            # Initial reconstruction: alpha_bar.sqrt() * ATy
            x = alpha_bar[-1].pow(0.5) * torch.cat([x_original, x_scaled], dim=1)
            x = RevBackProp.apply(x, self.body)

            main_out, aux_out = x.chunk(2, dim=1)
            main_out = main_out.view(b, S, c, h, w)
            aux_out = aux_out.view(b, S, c, h, w)

            merge_scales = self.merge_scale_factors.view(1, S, 1, 1, 1)
            result = main_out + merge_scales * aux_out
            result = result.reshape(b, sc, h, w)

            # fusion X to x
            return self.fusion(result)

        BaseNet.__init__ = __init__
        BaseNet.forward = forward
        BaseNet.unet_set_conv_io = unet_set_conv_io

        return BaseNet


class Net:
    def __new__(cls, T, unet):
        return My.Net()(T, unet)
