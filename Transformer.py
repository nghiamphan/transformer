import torch
import torch.nn as nn

from TransformerComponents import Embedding, PositionalEncoding, EncoderBlock


class Transformer(nn.Module):
    """
    A encoder-only Transformer.

    Arguments
    ---------
    src_vocab_size: int
        Vocabulary size of the input
    tgt_vocab_size: int
        Vocabulary size of the output
    embed_dim: int
        An embedding dimension of many layers of the transformer
    max_seq_length: int
        Maximum sequence length of an input
    n_heads: int
        Number of attention heads
    n_layers: int
        Number of encoder layers
    d_ff: int
        Dimension of the inner layer of the feed forward component
    dropout_rate: float
        Dropout rate applied on output of self attention and feed forward component
    apply_mask: bool
        Whether to apply mask on the input.
        If True, a token in the input sequence will only attend to non-padding tokens including itself.
        If False, it will attend to every token including padding tokens in the sequence.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int,
        max_seq_length: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout_rate: float,
        apply_mask: bool,
    ):
        super(Transformer, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.apply_mask = apply_mask

        self.embedding = Embedding(src_vocab_size, embed_dim, max_seq_length)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(embed_dim, n_heads, d_ff, dropout_rate) for i in range(n_layers)],
        )

        self.linear = nn.Linear(embed_dim, tgt_vocab_size)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        ---------
        src: torch.Tensor [batch_size, seq_length]

        Returns
        -------
        out: torch.Tensor [batch_size, max_seq_length, tgt_vocab_size]
        """

        # embedding layer pads each sequence with 0 to have length = max_seq_length
        # [batch_size, max_seq_length, embed_dim]
        src_embedding = self.embedding(src)
        # [batch_size, max_seq_length, embed_dim]
        src_embedding = self.positional_encoding(src_embedding)

        # [batch_size, max_seq_length, embed_dim]
        encode_output = src_embedding
        for encoder_layer in self.encoder_layers:
            if self.apply_mask:
                # [batch_size, 1, 1, seq_length]
                src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
            else:
                src_mask = None
            # [batch_size, max_seq_length, embed_dim]
            encode_output = encoder_layer(encode_output, mask=src_mask)

        # [batch_size, max_seq_length, embed_dim] -> [batch_size, max_seq_length, tgt_vocab_size]
        out = self.linear(encode_output)

        return out
