import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from CustomDataSet import CustomDataSet
from TransformerComponents import Embedding, PositionalEncoding, EncoderBlock, DecoderBlock


class VanillaTransformer(nn.Module):
    """
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
        Dropout rate for self attention and feed forward component
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
    ):
        super(VanillaTransformer, self).__init__()

        self.encoder_embedding = Embedding(src_vocab_size, embed_dim)
        self.decoder_embedding = Embedding(tgt_vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(embed_dim, n_heads, d_ff, dropout_rate) for i in range(n_layers)],
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(embed_dim, n_heads, d_ff, dropout_rate) for i in range(n_layers)],
        )

        self.linear = nn.Linear(embed_dim, tgt_vocab_size)

    def generate_mask(self, src, tgt):
        """
        Arguments
        ---------
        src: torch.Tensor [batch_size, seq_length]
        tgt: torch.Tensor [batch_size, seq_length]

        Returns
        -------
        src_mask: torch.Tensor [batch_size, 1, 1, seq_length]
        tgt_mask: torch.Tensor [1, seq_length, seq_length]
        """
        # [batch_size, 1, 1, seq_length]
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # [batch_size, 1, seq_length, 1]
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        seq_length = tgt.size(1)

        # [1, seq_length, seq_length]
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()

        # [batch_size, 1, seq_length, seq_length]
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        """
        Arguments
        ---------
        src: torch.Tensor [batch_size, seq_length]
        tgt: torch.Tensor [batch_size, seq_length]

        Returns
        -------
        out: torch.Tensor [batch_size, seq_length, tgt_vocab_size]
        """
        # src_mask [batch_size, 1, 1, seq_length]
        # tgt_mask [batch_size, 1, seq_length, seq_length]
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # [batch_size, seq_length, embed_dim]
        src_embedding = self.encoder_embedding(src)
        # [batch_size, seq_length, embed_dim]
        src_embedding = self.positional_encoding(src_embedding)

        # [batch_size, seq_length, embed_dim]
        tgt_embedding = self.decoder_embedding(tgt)
        # [batch_size, seq_length, embed_dim]
        tgt_embedding = self.positional_encoding(tgt_embedding)

        # [batch_size, seq_length, embed_dim]
        encode_output = src_embedding
        for encoder_layer in self.encoder_layers:
            # [batch_size, seq_length, embed_dim]
            encode_output = encoder_layer(encode_output, src_mask)

        # [batch_size, seq_length, embed_dim]
        decode_output = tgt_embedding
        for decode_layer in self.decoder_layers:
            # [batch_size, seq_length, embed_dim]
            decode_output = decode_layer(decode_output, encode_output, src_mask, tgt_mask)

        # [batch_size, seq_length, embed_dim] -> [batch_size, seq_length, tgt_vocab_size]
        out = self.linear(decode_output)

        return out


if __name__ == "__main__":
    # Set up data
    batch_size = 4
    dataset = CustomDataSet(max_seq_length=5, vocab_size=9)
    data_loader = DataLoader(dataset, batch_size, shuffle=True)

    # Set up model
    src_vocab_size = 10  # including padding token 0
    tgt_vocab_size = 10
    embed_dim = 512
    max_seq_length = 5
    n_heads = 8
    n_layers = 1
    d_ff = 2048
    dropout_rate = 0.1

    model = VanillaTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_dim=embed_dim,
        max_seq_length=max_seq_length,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
    )

    # Train model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    model.train()

    for epoch in range(10):
        for batch_input, batch_target in data_loader:
            optimizer.zero_grad()
            out = model(batch_input, batch_target)
            loss = criterion(out.view(-1, tgt_vocab_size), batch_target.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
