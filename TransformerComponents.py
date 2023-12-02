import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    """
    Arguments
    ---------
    vocab_size: int
        size of vocabulary
    embed_dim: int
        dimension of (word) embedding
    """

    def __init__(self, vocab_size: int, embed_dim: int, max_seq_length: int = None):
        super(Embedding, self).__init__()
        self.max_seq_length = max_seq_length
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        ---------
        x: torch.Tensor [batch_size, seq_length]
            The inner array is list of indices.
            Example: [[1, 2, 3],
                      [4, 5, 6]]

        Returns
        -------
        out: torch.Tensor [batch_size, seq_length, embed_dim]
            word embedding
        """
        # padding the sequence with 0 if necessary
        if self.max_seq_length != None:
            x = F.pad(x, (0, self.max_seq_length - x.size(1), 0, 0), value=0)

        out = self.embed(x)
        return out


class PositionalEncoding(nn.Module):
    """
    Arguments
    ---------
    embed_dim: int
        dimension of positional embedding, which is equal to dimension of word embedding
    max_seq_length: int
        the maximum length of an input sequence
    """

    def __init__(self, embed_dim: int, max_seq_length: int):
        super(PositionalEncoding, self).__init__()

        # [max_seq_length, embed_dim]
        pe = torch.zeros(max_seq_length, embed_dim)

        # [max_seq_length, 1]
        position = torch.arange(max_seq_length).unsqueeze(1)

        # [1, (embed_dim+1) // 2]
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000) / embed_dim))

        # [max_seq_length, (embed_dim+1) // 2]
        pe[:, 0::2] = torch.sin(position * div_term)
        # [max_seq_length, embed_dim // 2]
        pe[:, 1::2] = torch.cos(position * div_term)

        # [1, max_seq_length, embed_dim]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        ---------
        x: torch.Tensor [batch_size, seq_length, embed_dim]
            word embedding

        Returns
        -------
        out: torch.Tensor [batch_size, seq_length, embed_dim]
            word embedding + positional encoding
        """
        # [batch_size, seq_length, embed_dim]
        out = x + self.pe[:, : x.size(1)]
        return out


class MultiHeadAttention(nn.Module):
    """
    Arguments
    ---------
    embed_dim: int
        dimension of input and output of MultiHeadAttention component
        Note: this dimension does not need to be the same as dimension of word embedding,
        but in practice we usually set them to be the same to reduce the number of parameters
        in the transformer model.
    n_heads: int
        number of self attention heads
    """

    def __init__(self, embed_dim: int, n_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % n_heads == 0

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        # Dimension of each head's key, query, value
        self.d_k = embed_dim // n_heads

        # key, query, value and output matrices
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Arguments
        ---------
        Q: torch.Tensor [batch_size, n_heads, seq_length, d_k]
        K: torch.Tensor [batch_size, n_heads, seq_length, d_k]
        V: torch.Tensor [batch_size, n_heads, seq_length, d_k]
        mask: torch.Tensor
            encoder mask's dimension: [batch_size, 1, 1, seq_length]
            decoder mask's dimension: [1, seq_length, seq_length]

        Returns
        -------
        out: torch.Tensor [batch_size, n_heads, seq_length, d_k]
        """
        # [batch_size, n_heads, seq_length, d_k] * [batch_size, n_heads, d_k, seq_length] = [batch_size, n_heads, seq_length, seq_length]
        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # apply mask if provided
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e20)

        attention_score = torch.softmax(attention_score, dim=-1)

        # [batch_size, n_heads, seq_length, seq_length] * [batch_size, n_heads, seq_length, d_k] = [batch_size, n_heads, seq_length, d_k]
        attention_score = torch.matmul(attention_score, V)
        return attention_score

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        ---------
        x: torch.Tensor [batch_size, seq_length, embed_dim]

        Returns
        -------
        out: torch.Tensor [batch_size, n_heads, seq_length, d_k]
        """
        batch_size, seq_length, embed_dim = x.size()
        # [batch_size, n_heads, seq_length, d_k]
        out = x.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        return out

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        ---------
        x: torch.Tensor [batch_size, n_heads, seq_length, d_k]

        Returns
        -------
        out: torch.Tensor [batch_size, seq_length, embed_dim]
        """
        batch_size, n_heads, seq_length, d_k = x.size()
        # [batch_size, n_heads, seq_length, d_k] -> [batch_size, seq_length, embed_dim]
        out = x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        return out

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Arguments
        ---------
        x_q: torch.Tensor [batch_size, seq_length, embed_dim]
            vector to be projected into a query vector by learned matrix W_q
        x_k: torch.Tensor [batch_size, seq_length, embed_dim]
            vector to be projected into a key vector by learned matrix W_k
        x_v: torch.Tensor [batch_size, seq_length, embed_dim]
            vector to be projected into a value vector by learned matrix W_v
        mask: torch.Tensor
            encoder mask's dimension: [batch_size, 1, 1, seq_length]
            decoder mask's dimension: [1, seq_length, seq_length]

        Returns
        -------
        out: torch.Tensor [batch_size, seq_length, embed_dim]
        """
        # [batch_size, seq_length, embed_dim] -> [batch_size, seq_length, embed_dim]
        Q = self.W_q(x_q)
        # [batch_size, seq_length, embed_dim] -> [batch_size, n_heads, seq_length, d_k]
        Q = self.split_heads(Q)
        K = self.split_heads(self.W_k(x_k))
        V = self.split_heads(self.W_v(x_v))

        # [batch_size, n_heads, seq_length, d_k]
        out = self.scaled_dot_product_attention(Q, K, V, mask)

        # [batch_size, n_heads, seq_length, d_k] -> [batch_size, seq_length, embed_dim]
        out = self.combine_heads(out)
        out = self.W_o(out)
        return out


class FeedForward(nn.Module):
    """
    Arguments:
    embed_dim: int
        dimension of input and output layer of FeedForward component
    d_ff: int
        dimension of inner layer
    """

    def __init__(self, embed_dim: int, d_ff: int):
        super(FeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
        x: torch.Tensor [batch_size, seq_length, embed_dim]

        Returns
        -------
        out: torch.Tensor [batch_size, seq_length, embed_dim]
        """
        out = self.feed_forward(x)
        return out


class EncoderBlock(nn.Module):
    """
    Arguments
    ---------
    embed_dim: int
        dimension of input and output of EncodeBlock component
    n_heads: int
        number of attention heads
    d_ff: int
        dimension of inner layer of feed forward componenet
    dropout_rate: float
        dropout rate applied to self_attention and feed forward layer
    """

    def __init__(self, embed_dim: int, n_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super(EncoderBlock, self).__init__()

        self.self_attention = MultiHeadAttention(embed_dim, n_heads)
        self.feed_forward = FeedForward(embed_dim, d_ff)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Arguments:
        x: torch.Tensor [batch_size, seq_length, embed_dim]
        mask: torch.Tensor [batch_size, 1, 1, seq_length]

        Returns
        -------
        out: torch.Tensor [batch_size, seq_length, embed_dim]
        """
        # [batch_size, seq_length, embed_dim]
        self_attention_output = self.self_attention(x, x, x, mask)
        # [batch_size, seq_length, embed_dim]
        out = self.norm1(x + self.dropout(self_attention_output))

        # [batch_size, seq_length, embed_dim] -> [batch_size, seq_length, d_ff] -> [batch_size, seq_length, embed_dim]
        feed_forward_output = self.feed_forward(out)

        # [batch_size, seq_length, embed_dim]
        out = self.norm2(out + self.dropout(feed_forward_output))
        return out


class DecoderBlock(nn.Module):
    """
    Arguments
    ---------
    embed_dim: int
        dimension of input and output of DecoderBlock component
    n_heads: int
        number of attention heads
    d_ff: int
        dimension of inner layer of feed forward componenet
    dropout_rate: float
        dropout rate applied to self attention, cross attention and feed forward layer
    """

    def __init__(self, embed_dim: int, n_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super(DecoderBlock, self).__init__()

        self.self_attention = MultiHeadAttention(embed_dim, n_heads)
        self.cross_attention = MultiHeadAttention(embed_dim, n_heads)
        self.feed_forward = FeedForward(embed_dim, d_ff)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        Arguments:
        x: torch.Tensor [batch_size, seq_length, embed_dim]
        enc_output: torch.Tensor [batch_size, seq_length, embed_dim]
        src_mask: torch.Tensor [batch_size, 1, 1, seq_length]
        tgt_mask: torch.Tensor [1, seq_length, seq_length]

        Returns
        -------
        out: torch.Tensor [batch_size, seq_length, embed_dim]
        """
        self_attention_output = self.self_attention(x, x, x, tgt_mask)
        out = self.norm1(x + self.dropout(self_attention_output))

        cross_attention_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        out = self.norm2(out + self.dropout(cross_attention_output))

        feed_forward_output = self.feed_forward(out)
        out = self.norm3(out + self.dropout(feed_forward_output))
        return out
