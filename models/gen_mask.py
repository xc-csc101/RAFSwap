import torch
from torch import nn
import math
from models.stylegan2.model import ConvLayer

class Mask(nn.Module):

    def __init__(self, channel_multiplier=2, size=256):
        super(Mask, self).__init__()

        channels = {
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
        }

        self.log_size = (int(math.log(size, 2)) - 1) * 2
        self.convs = nn.ModuleList()

        for i in range(2, self.log_size):
            idx = i // 2
            s = 2 ** (idx + 2)
            self.convs.append(ConvLayer(channels[s], 32, 3))
            scale = size // s
            self.convs.append(nn.Upsample(scale_factor=scale))
        
        self.to_mask = nn.Sequential(*[
            ConvLayer(32 * 12, 128, 3),
            ConvLayer(128, 1, 3, activate=False),
            nn.Sigmoid()
        ])
        
    
    def forward(self, feats):
        feats = feats[2:-4]
        feats_rs = []

        for i, (conv, up) in enumerate(zip(self.convs[::2], self.convs[1::2])):
            feats_rs.append(up(conv(feats[i])))
        
        mask = self.to_mask(torch.cat(feats_rs, dim=1))
        return mask
    