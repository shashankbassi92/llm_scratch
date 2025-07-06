import torch.nn as nn
import torch


class SelfAttention(nn.Module):
    """
    Implementation of vanilla Self Attention module. Main part of transformer architecture.
    qkv_bias could be optionally turned on. GPT architecture 
    keeps it off.
    nn.Linear is better than nn.Parameter for weight matrices as it provides better
    initialization.
    """
    def __init__(self, din, dout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(din, dout, bias=qkv_bias)
        self.W_key = nn.Linear(din, dout, bias=qkv_bias)
        self.W_value = nn.Linear(din, dout, bias=qkv_bias)

    def forward(self, x):
        query = self.W_query(x)                 # out = (bs, context_len, dout)
        key = self.W_key(x)                     # out = (bs, context_len, dout)
        value = self.W_value(x)                 # out = (bs, context_len, dout)

        # (bs, context_len, dout) * (bs, dout, context_len), 
        # out = (bs, context_len, context_len)
        attention_scores = query @ key.transpose(1,2) 
        attention_weights = torch.softmax(
            attention_scores/key.shape[-1]**0.5,
            dim=-1
        )                                       # normalize scores to get weights
        context = attention_weights @ value     # weighted addition of scaled input values
        return context

class CausalAttention(nn.Module):
    """
    Decoder only models generally use causal attention to mask
    tokens which come afterwards in the sequence. So, causal
    attention is the solution for that.
    """
    def __init__(self, din, dout, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(din, dout, bias=qkv_bias)
        self.W_key = nn.Linear(din, dout, bias=qkv_bias)
        self.W_value = nn.Linear(din, dout, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # register_buffer for not making this a learnable param
        # diagonal=1 to strictly not attend query token itself
        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    def forward(self, x):
        query = self.W_query(x)                 # out = (bs, context_len, dout)
        key = self.W_key(x)                     # out = (bs, context_len, dout)
        value = self.W_value(x)                 # out = (bs, context_len, dout)
        
        # (bs, context_len, dout) * (bs, dout, context_len), 
        # out = (bs, context_len, context_len)
        attention_scores = query @ key.transpose(1,2) 
        # trick to softmax with -inf and output 0
        attention_scores.masked_fill_(  
            self.mask.bool(), -torch.inf
        )
        attention_weights = torch.softmax(
            attention_scores/key.shape[-1]**0.5,
            dim=-1
        )                                       # normalize scores to get weights
        attention_weights = self.dropout(attention_weights)
        context = attention_weights @ value     # weighted addition of scaled input values
        return context



# class MultiHeadAttention(nn.Module):


# class LayerNorm(nn.Module):


# class FeedForwardNetwork(nn.Module):


# class TransformerBlock(nn.Module):


import unittest
class TestAttentionModule(unittest.TestCase):

    def test_selfAttention(self):
        atm = SelfAttention(din=4, dout=2)
        x = torch.randn(3, 6, 4)
        out = atm(x)
        self.assertEqual(out.shape, (3, 6, 2))

    def test_causalAttention(self):
        cas = CausalAttention(din=4, dout=2, context_length=6, dropout=0.3)
        x = torch.randn(3, 6, 4)
        out = cas(x)
        self.assertEqual(out.shape, (3, 6, 2))


if __name__=='__main__':
    unittest.main()

    