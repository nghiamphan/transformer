import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from TransformerComponents import Embedding, PositionalEncoding, EncoderBlock, DecoderBlock
from config import SAMPLE_SIZE_BY_SEQ_LENGTH, MAX_SEQ_LENGTH, VOCAB_SIZE, RANDOM_STATE


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
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.tgt_vocab_size = tgt_vocab_size

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
        src: torch.Tensor [batch_size, src_seq_length]
        tgt: torch.Tensor [batch_size, tgt_seq_length]

        Returns
        -------
        src_mask: torch.Tensor [batch_size, 1, 1, src_seq_length]
        tgt_mask: torch.Tensor [1, tgt_seq_length, tgt_seq_length]
        """
        # [batch_size, 1, 1, src_seq_length]
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        # [batch_size, 1, tgt_seq_length, 1]
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        tgt_seq_length = tgt.size(1)

        # [1, tgt_seq_length, tgt_seq_length]
        nopeak_mask = (1 - torch.triu(torch.ones(1, tgt_seq_length, tgt_seq_length), diagonal=1)).bool().to(self.device)

        # [batch_size, 1, tgt_seq_length, tgt_seq_length]
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        """
        Arguments
        ---------
        src: torch.Tensor [batch_size, src_seq_length]
        tgt: torch.Tensor [batch_size, tgt_seq_length]

        Returns
        -------
        out: torch.Tensor [batch_size, tgt_seq_length, tgt_vocab_size]
        """

        """
        Let say
        src = [[1, 2, 3, 0, 0],
               [4, 5, 6, 7, 9]]
        effective_lengths = [3, 5]
        """
        effective_lengths = (src != 0).sum(dim=1).tolist()

        # src_mask [batch_size, 1, 1, src_seq_length]
        # tgt_mask [batch_size, 1, tgt_seq_length, tgt_seq_length]
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # [batch_size, src_seq_length, embed_dim]
        src_embedding = self.encoder_embedding(src)
        # [batch_size, src_seq_length, embed_dim]
        src_embedding = self.positional_encoding(src_embedding)

        # [batch_size, tgt_seq_length, embed_dim]
        tgt_embedding = self.decoder_embedding(tgt)
        # [batch_size, tgt_seq_length, embed_dim]
        tgt_embedding = self.positional_encoding(tgt_embedding)

        # [batch_size, src_seq_length, embed_dim]
        encode_output = src_embedding
        for encoder_layer in self.encoder_layers:
            # [batch_size, src_seq_length, embed_dim]
            encode_output = encoder_layer(encode_output, src_mask)

        # [batch_size, tgt_seq_length, embed_dim]
        decode_output = tgt_embedding
        for decode_layer in self.decoder_layers:
            # [batch_size, tgt_seq_length, embed_dim]
            decode_output = decode_layer(decode_output, encode_output, src_mask, tgt_mask, effective_lengths)

        # [batch_size, tgt_seq_length, embed_dim] -> [batch_size, tgt_seq_length, tgt_vocab_size]
        out = self.linear(decode_output)
        return out

    def model_training(
        self, train_data_loader: DataLoader, epochs: int = 10, lr: int = 1e-5, print_loss: bool = False
    ) -> float:
        """
        Arguments
        ---------
        train_data_loader: torch.utils.data.DataLoader
        epochs: int
        lr: int
        print_loss: bool

        Returns
        -------
        loss: float
            cross entropy loss
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-5)

        self.to(self.device)
        self.train()

        for epoch in range(epochs):
            for batch_input, batch_target in train_data_loader:
                batch_input = batch_input.to(self.device)
                batch_target = batch_target.to(self.device)
                batch_decoder_input = batch_target[:, :-1]
                batch_real_target = batch_target[:, 1:]

                optimizer.zero_grad()
                out = self(batch_input, batch_decoder_input)
                loss = criterion(
                    out.view(-1, self.tgt_vocab_size),
                    batch_real_target.contiguous().view(-1),
                )
                loss.backward()
                optimizer.step()
            if print_loss:
                print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

        return loss

    def model_eval(self, input_test: torch.Tensor, target_test: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Arguments
        ---------
        input_test: torch.Tensor [batch_size, src_seq_length]
        target_test: torch.Tensor [batch_size, src_seq_length + 1, tgt_vocab_size]

        Returns
        -------
        out_test: torch.Tensor [batch_size, src_seq_length, tgt_vocab_size]
        cross_entropy_loss: float
        """
        out_probs = self.generate_output(input_test)

        real_batch_target = target_test[:, 1:].to(self.device)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(
            out_probs.view(-1, self.tgt_vocab_size),
            real_batch_target.contiguous().view(-1),
        )
        cross_entropy_loss = loss.item()

        return out_probs, cross_entropy_loss

    def generate_output(self, input_test: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        ---------
        input_test: torch.Tensor [batch_size, src_seq_length]

        Returns
        -------
        out_probs: torch.Tensor [batch_size, src_seq_length, tgt_vocab_size]
        """
        input_test = input_test.to(self.device)
        self.to(self.device)
        self.eval()

        # [batch_size, 1, tgt_vocab_size]
        start_token_probs = torch.zeros(input_test.size(0), 1, self.tgt_vocab_size, device=self.device)
        start_token_probs[:, :, -1] = 1

        with torch.no_grad():
            decoder_input_probs = start_token_probs
            # generate output until the length of the output is equal to the length of the input
            while decoder_input_probs.size(1) <= input_test.size(1):
                out_probs = self(input_test, torch.argmax(decoder_input_probs, dim=-1))
                decoder_input_probs = torch.cat((start_token_probs, out_probs), dim=1)

        return out_probs


def objective(
    trial,
    train_data_loader,
    input_val: torch.Tensor,
    target_val: torch.Tensor,
    vocab_size: int,
    max_seq_length: int,
    epochs: int,
) -> float:
    """
    Tune Transformer model's parameters to minimize cross entropy loss.
    """
    src_vocab_size = vocab_size + 1  # adding padding token 0
    tgt_vocab_size = vocab_size + 2  # adding padding token 0 and start token
    embed_dim = trial.suggest_categorical("embed_dim", [256, 512, 1024, 2048])
    max_seq_length = max_seq_length
    n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8])
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 4])
    d_ff = trial.suggest_categorical("d_ff", [512, 1024, 2048, 4096])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0, 0.1, 0.2, 0.3, 0.4, 0.5])

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

    model.model_training(train_data_loader, epochs, lr=1e-5)
    cross_entropy_loss = model.model_eval(input_val, target_val)[1]
    return cross_entropy_loss


def model_tuning(
    train_data_loader,
    input_val: torch.Tensor,
    target_val: torch.Tensor,
    vocab_size: int,
    max_seq_length: int,
    epochs: int,
    n_trials: int,
) -> dict:
    """
    Arguments
    ---------
    train_data_loader: torch.utils.data.DataLoader
    input_val: torch.Tensor
    target_val: torch.Tensor
    vocab_size: int
    max_seq_length: int
    epochs: int
    n_trials: int

    Returns
    -------
    study.best_params: dict
    """
    study = optuna.create_study()
    study.optimize(
        lambda trial: objective(trial, train_data_loader, input_val, target_val, vocab_size, max_seq_length, epochs),
        n_trials=n_trials,
    )

    return study.best_params


class CustomDataSet(Dataset):
    """
    Arguments
    ---------
    sample_size_by_seq_length: dict[seq_length: sample_size]
        This is a dictionary where a key is a sequence length and its value is the number
        of sequences to be generated by that sequence length.
        Example: {1: 4, 2: 20, 3: 100, 4: 400, 5: 1000}
    max_seq_length: int
        the maximum length of each sequence
    vocab_size: int
        the size of vocabulary to be generated
    random_state: int
    """

    def __init__(
        self,
        sample_size_by_seq_length: dict = SAMPLE_SIZE_BY_SEQ_LENGTH,
        max_seq_length: int = MAX_SEQ_LENGTH,
        vocab_size: int = VOCAB_SIZE,
        random_state: int = RANDOM_STATE,
    ):
        torch.manual_seed(random_state)

        self.input = torch.empty(0, max_seq_length, dtype=torch.int)
        self.target = torch.empty(0, max_seq_length + 1, dtype=torch.int)

        for seq_length, sample_size in sample_size_by_seq_length.items():
            # Dimension of input and target: [sample_size, max_seq_length]
            input, target = self.generate_sample(
                sample_size,
                max_seq_length,
                seq_length,
                vocab_size,
            )
            self.input = torch.cat((self.input, input), dim=0)
            self.target = torch.cat((self.target, target), dim=0)

    def generate_sample(self, sample_size: int, max_seq_length: int, seq_length: int, vocab_size: int):
        """
        Example:
        Let seq_length = 3, max_seq_length = 5, vocab_size = 9 (meaning the actual vocabulary is from 1 to 9)
        Suppose the initial generated input = [1, 2, 3]
        Then the padded input = [1, 2, 3, 0, 0]
        Then the target = [10, 3, 2, 1, 0, 0]
        where 10 is the start token and 0 is the padding token.

        Arguments
        ---------
        sample_size: int
            number of sequences to generate
        max_seq_length: int
            the maximum length of each sequence
        seq_length: int
            the length of each sequence. If seq_length < max_seq_length, each sequence will be padded with 0 at the end.
        vocab_size: int
            the size of vocabulary to be generated (not including padding token 0)

        Returns
        -------
        input: torch.Tensor [sample_size, max_seq_length]
        target: torch.Tensor [sample_size, max_seq_length + 1]
        """
        # generate input sequences consisting of numbers from 1 to vocab_size
        # [sample_size, seq_length]
        input = torch.randint(low=1, high=vocab_size + 1, size=(sample_size, seq_length))

        # keep only uniquely generated inputs
        input = torch.unique(input, dim=0)

        # generate target sequences which are reverses of input sequences
        # [sample_size, seq_length]
        target = torch.flip(input, dims=(1,))

        # pad the input and target at the end with 0s so that each sequence has length = max_seq_length
        # [sample_size, max_seq_length]
        input = F.pad(input, (0, max_seq_length - seq_length, 0, 0), value=0)
        target = F.pad(target, (0, max_seq_length - seq_length, 0, 0), value=0)

        # add a column of (vocab_size)s to the beginning of target maatrix
        # [sample_size, max_seq_length + 1]
        target = torch.cat(
            (torch.full((target.size(0), 1), vocab_size + 1), target),
            dim=1,
        )

        return input, target

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input = self.input[index]
        target = self.target[index]
        return input, target
