import torch.nn as nn
import copy
import sys
sys.path.append("..")
#from utils.subla import SublayerConnection, PositionwiseFeedForward
from utils.sublayer import SublayerConnection
from utils.layer_norm import LayerNorm

from utils.feed_forward import PositionwiseFeedForward

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

