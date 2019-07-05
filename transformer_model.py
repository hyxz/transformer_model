import torch.nn as nn

import copy
from attention.multi_head import MultiHeadedAttention
from utils.sublayer import SublayerConnection
from utils.position_encoder import PositionalEncoding
from utils.feed_forward import PositionwiseFeedForward
from en_decoder.en_decoer import EncoderDecoder
from en_decoder.encoder import Encoder
from en_decoder.decoder import Decoder
from en_decoder.decoder_layer import DecoderLayer
from en_decoder.encoder_layer import EncoderLayer
from utils.embeddings import Embeddings
from en_decoder.generation import Generator

#from .attention import MultiHeadedAttention
#from .utils import SublayerConnection, PositionwiseFeedForward,PositionalEncoding
#from .en_decoder import EncoderDecoder
#from .en_decoder import Encoder,Decoder

def trans_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    """

    :param src_vocab: src vocab
    :param tgt_vocab:
    :param N: number of
    :param d_model:
    :param d_ff:
    :param h: hidden layers
    :param dropout:
    :return: 
    """
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))


    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

if __name__ == "__main__":
    # test example model.
    tmp_model = trans_model(10, 10)