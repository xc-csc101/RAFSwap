
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class Projector(nn.Module):
    def __init__(self, in_channels, out_channels, token_channels, cls=4):
        super(Projector, self).__init__()
        self.cls = cls
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.token_channels = token_channels
        
        self.linear1 = nn.Linear(in_channels, token_channels, bias=False) 
        self.linear2 = nn.Linear(token_channels , token_channels, bias=False) 
        self.linear3 = nn.Linear(token_channels, out_channels) 
        
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.xavier_normal_(self.linear3.weight)
 
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Linear(in_channels, out_channels),
            )
        
    def forward_cell(self, x: Tensor, t: Tensor) -> Tensor:  
        N, HW, C = x.shape
        _, L, _ = t.shape
        x_q = self.linear1(x) 
        t_q = self.linear2(t) 

        t_q = torch.transpose(t_q, 1, 2) 
        a = x_q.matmul(t_q) 

        with torch.no_grad():
            v = a.detach().nonzero().long().permute(1, 0)
            weight_ind = v.clone()
            del v
            torch.cuda.empty_cache()
        
        a = F.softmax(a / np.sqrt(C), dim=2)
        a = a[weight_ind[0], weight_ind[1], weight_ind[2]]

        a = torch.sparse.FloatTensor(weight_ind, a, torch.Size([N, HW, L]))
        a = a.to_dense()

        t = self.linear3(t) 

        a = a.matmul(t)

        return a
    
    def forward(self, mask_t, fea_t, token_s):
        bs, channel_num, H, W = fea_t.shape

        mask_t_re = mask_t.view(-1, *mask_t.shape[2:])
        mask_t_re = F.interpolate(mask_t_re, size=(H, W)).repeat(1, channel_num, 1, 1) 
        fea_c = fea_t.repeat(self.cls, 1, 1, 1)            
        fea_c = fea_c * mask_t_re
        fea_c = fea_c.view(bs, self.cls, channel_num, H, W)
        fea_c = torch.mean(fea_c, dim=1)

        q = fea_c.view(*fea_c.shape[:2], -1)
        q = q.permute(0, 2, 1)
        t = token_s
        
        out = self.forward_cell(q, t).permute(0, 2, 1)
        out = out.view(bs, channel_num, H, W)
        out = out + fea_t
        return out