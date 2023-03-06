import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
from models.stylegan2.model import EqualLinear
from models.projector import Projector
from models.transformer import Block

class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x

class Tokenizer(nn.Module):
    def __init__(self, inchl):
        super(Tokenizer, self).__init__()
        self.fc = nn.Sequential(*[nn.Linear(inchl, 512)])

    def forward(self, codes, segmap):
        b, c, [h, w] = segmap.shape[0], segmap.shape[1], segmap.shape[3:]
        segmap = segmap.view(b, c, h, w)

        segmap = F.interpolate(segmap, size=codes.size()[2:], mode='nearest')

        b_size = codes.shape[0]
        f_size = codes.shape[1]

        s_size = segmap.shape[1]

        codes_vector = torch.zeros((b_size, s_size, f_size), dtype=codes.dtype, device=codes.device)

        for i in range(b_size):
            for j in range(s_size):
                component_mask_area = torch.sum(segmap.bool()[i, j])

                if component_mask_area > 0:
                    codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]).reshape(f_size,  component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature

        return self.fc(codes_vector)

class GradualStyleSwapEncoder(Module):
    'complete implemantion'
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleSwapEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = 18
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.avg = nn.AdaptiveAvgPool2d(1)
        # self.glb = nn.Sequential(*[EqualLinear(512, 512, lr_mul=1),
        #                            EqualLinear(512, 512, lr_mul=1)])

        self.token = nn.Sequential(*[Tokenizer(128), Tokenizer(256), Tokenizer(512)])
        self.proj = nn.Sequential(*[Projector(512, 512, 512), Projector(512, 512, 512), Projector(512, 512, 512)])

        self.trans = nn.ModuleList()
        self.depth = opts.depth
        for _ in range(self.depth):
            self.trans.append(Block(dim=512, num_heads=8))

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x, mask_t, mask_s):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x
        
        c1t, c1s = torch.chunk(c1, 2, dim=0)
        c2t, c2s = torch.chunk(c2, 2, dim=0)
        c3t, c3s = torch.chunk(c3, 2, dim=0)

        c3s_avg = self.avg(c3s)
        # c3s_avg = self.glb(self.avg(c3s).flatten(1)).unsqueeze(-1).unsqueeze(-1)

        c1s, c2s, c3s = self.token[0](c1s, mask_s), self.token[1](c2s, mask_s), self.token[2](c3s, mask_s)

        cs = torch.cat([c1s, c2s, c3s], dim=1)

        for i in range(self.depth):
            cs = self.trans[i](cs)
        c1s, c2s, c3s = cs[:, :4], cs[:, 4:8], cs[:, 8:12]

        c3 = self.proj[2](mask_t, c3t, c3s)
        c3 = c3 + c3s_avg.expand_as(c3)
        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        # p2 = self._upsample_add(c3, self.latlayer1(c2t))
        p2 = self.latlayer1(c2t)
        p2 = self.proj[1](mask_t, p2, c2s)
        p2 = p2 + c3s_avg.expand_as(p2)

        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        # p1 = self._upsample_add(p2, self.latlayer2(c1t))
        p1 = self.latlayer2(c1t)
        p1 = self.proj[0](mask_t, p1, c1s)
        p1 = p1 + c3s_avg.expand_as(p1)

        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x

class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.n_styles = opts.n_styles
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * self.n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, self.n_styles, 512)
        return x