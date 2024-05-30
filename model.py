import dgl.nn.pytorch as dglnn
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.utils import expand_as_pair


class SAGNN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(SALayer(in_size, hid_size, hid_size))
            else:
                self.layers.append(SALayer(hid_size, hid_size, hid_size))
        self.linear = nn.Linear(hid_size, out_size)

    def forward(self, blocks, h):
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return self.linear(h)

class SALayer(nn.Module):
    def __init__(self, in_size, hid_size, out_size, activation=F.relu):
        super().__init__()

        self.activation = activation
        self.src_proj = dglnn.linear.TypedLinear(in_size, hid_size, num_types=2, regularizer='basis', num_bases=1)
        self.dst_proj = dglnn.linear.TypedLinear(in_size, hid_size, num_types=2, regularizer='basis', num_bases=1)
        self.linear = nn.Linear(in_size + hid_size, out_size)
        self.W = nn.Linear(in_size, out_size)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        nn.init.constant_(self.linear.bias, 0.)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.W.bias, 0.)

    def message(self, edges):
        src_feat = self.src_proj(edges.src['n'], edges.data['first_etype'])
        feat = self.dst_proj(edges.dst['n'], edges.data['second_etype'])
        feat = self.activation(torch.cat([src_feat, feat], dim=-1))
        m = feat

        return {'m': m}

    def forward(self, g, h):
        h_src, h_dst = expand_as_pair(h, g)
        with g.local_scope():
            g.srcdata['n'] = h_src
            g.dstdata['n'] = h_dst

            g.update_all(self.message, fn.max('m', 'n'))

            n = g.dstdata['n']
            z = self.activation(self.linear(n) + self.W(h_dst))

            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm

            return z
