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
        query = self.W_query(x)  # out = (bs, context_len, dout)
        key = self.W_key(x)  # out = (bs, context_len, dout)
        value = self.W_value(x)  # out = (bs, context_len, dout)

        # (bs, context_len, dout) * (bs, dout, context_len),
        # out = (bs, context_len, context_len)
        attention_scores = query @ key.transpose(1, 2)
        attention_weights = torch.softmax(
            attention_scores / key.shape[-1] ** 0.5, dim=-1
        )  # normalize scores to get weights
        context = attention_weights @ value  # weighted addition of scaled input values
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
        # also if we just use a plan tensor then its not a param like self.W_query.weight.
        # so, if you are moving this class/model to another device then mask wont move automatically.
        # diagonal=1 to mask future tokens and not itself (if diagonal=0)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        query = self.W_query(x)  # out = (bs, context_len, dout)
        key = self.W_key(x)  # out = (bs, context_len, dout)
        value = self.W_value(x)  # out = (bs, context_len, dout)

        # (bs, context_len, dout) * (bs, dout, context_len),
        # out = (bs, context_len, context_len)
        attention_scores = query @ key.transpose(1, 2)
        # trick to softmax with -inf and output 0
        attention_scores.masked_fill_(self.mask.bool(), -torch.inf)  # in-place
        attention_weights = torch.softmax(
            attention_scores / key.shape[-1] ** 0.5, dim=-1
        )  # normalize scores to get weights
        attention_weights = self.dropout(attention_weights)
        # weighted addition of scaled input values
        # out = (bs, context_len, dout)
        context = attention_weights @ value
        return context


class MultiHeadAttention(nn.Module):
    """
    Concatenate multiple outputs from parallel attention paths
    """

    def __init__(self, din, dout, context_length, dropout, num_heads, qkv_bias=False):
        """
        d_out: length of final concatenated multi-head output
        """
        super().__init__()
        assert (
            dout % num_heads == 0
        ), "To make sure final length is divisible by num-heads for symmetry of each head."
        self.head_dim = dout // num_heads
        self.dout = dout
        self.num_heads = num_heads

        self.W_query = nn.Linear(din, dout, bias=qkv_bias)
        self.W_key = nn.Linear(din, dout, bias=qkv_bias)
        self.W_value = nn.Linear(din, dout, bias=qkv_bias)
        self.out_proj = nn.Linear(
            dout, dout
        )  # Linear last layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, context_len, d_in = x.shape
        queries = self.W_query(x)  # out = (bs, context_len, dout)
        keys = self.W_key(x)  # out = (bs, context_len, dout)
        values = self.W_value(x)  # out = (bs, context_len, dout)

        # create new dimension for num_heads
        # shape: (bs, context_len, num_heads, dout_head)
        keys = keys.view(b, context_len, self.num_heads, self.head_dim)
        values = values.view(b, context_len, self.num_heads, self.head_dim)
        queries = queries.view(b, context_len, self.num_heads, self.head_dim)

        # transpose to bring up num_heads
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # create attention weights of context_len * context len
        attention_scores = queries @ keys.transpose(2, 3)
        attention_scores.masked_fill(self.mask.bool(), -torch.inf)  # in-place
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        # dot product with value to create context_len
        # out: (bs, num_heads, context_len, dout)
        context = attention_weights @ values

        # transpose back to (bs, context_len, num_heads, dout)
        context = context.transpose(1, 2)

        # reshape (bs, context_len, -1)
        context = context.contiguous().view(b, context_len, self.dout)
        context = self.out_proj(context)
        return context


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

    def test_multiHeadAttention(self):
        mha = MultiHeadAttention(4, 8, 6, 0.3, num_heads=2)
        x = torch.randn(3, 6, 4)
        out = mha(x)
        self.assertEqual(out.shape, (3, 6, 8))


if __name__ == "__main__":
    unittest.main()
