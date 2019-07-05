"""
encde
"""
import torch.nn as nn
import copy
import sys
sys.path.append("..")
#sys.path.append("/home/zhouxin/xzworkspace/bert-pyorch_s/bert_pytorch/")
from utils.sublayer import SublayerConnection
#from ..utils.sublayer import SublayerConnection
from utils.position_encoder import PositionalEncoding
from utils.layer_norm import LayerNorm
from utils.feed_forward import PositionwiseFeedForward

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)