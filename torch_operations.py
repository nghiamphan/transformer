import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from CustomDataSet import CustomDataSet
from Transformer import Transformer
from model_utils import model_training, model_eval, model_tuning
from config import SAMPLE_SIZE_BY_SEQ_LENGTH, MAX_SEQ_LENGTH, VOCAB_SIZE, RANDOM_STATE


def test_argmax():
    x = torch.tensor([[1, 2, 3], [4, 8, 6]])
    out = x.argmax(dim=1)
    print("Original tensor:")
    print(x)
    print("Indices of max values along dimension 1:")
    print(out)


def test_mask():
    src = torch.tensor([[1, 2, 0], [4, 5, 6]])
    tgt = torch.tensor([[4, 5, 0], [2, 3, 0]])

    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

    print("src_mask")
    print(src_mask.shape)
    print(src_mask)
    print("\ntgt_mask")
    print(tgt_mask.shape)
    print(tgt_mask)

    seq_length = tgt.size(1)
    print("\nseq_length", seq_length)

    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    print("\nnopeak_mask")
    print(nopeak_mask.shape)
    print(nopeak_mask)

    tgt_mask = tgt_mask & nopeak_mask
    print("\ntgt_mask")
    print(tgt_mask.shape)
    print(tgt_mask)


def test_unsqueeze():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print("x original")
    print(x)

    x = x.unsqueeze(1)
    print("\nx")
    print(x)


def test_cross_entrophy_loss():
    pred = torch.tensor([[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]])
    print("pred.shape", pred.shape)

    target = torch.tensor([[1, 1], [1, 0]])
    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(pred.view(-1, 3), target.view(-1))
    print(loss)


def test_cat():
    # [2, 1, 3]
    input = torch.tensor([[0, 1, 2], [1, 2, 3]]).unsqueeze(1)
    target = torch.tensor([[5, 6, 7], [4, 5, 6]]).unsqueeze(1)

    # Concatenate along dimension 1 (columns)
    # [2, 2, 3]
    appended_tensor = torch.cat((input, target), dim=1)

    empty = torch.empty(0, 2, 3)
    data = torch.cat((empty, appended_tensor), dim=0)

    print("Original Tensor 1:")
    print(input)

    print("\nOriginal Tensor 2:")
    print(target)

    print("\nAppended Tensor:")
    print(appended_tensor)

    print("\nData Tensor:")
    print(data)


def test_layer_norm():
    input_tensor = torch.randn(3, 4, 5)  # Example shape (batch_size, sequence_length, input_size)
    print(input_tensor)

    # LayerNorm along the last dimension (input_size)
    layer_norm = nn.LayerNorm(5)

    # Apply LayerNorm to the input tensor
    output_tensor = layer_norm(input_tensor)
    print(output_tensor)


def test_optuna():
    # Set up data
    sample_size_by_seq_length = SAMPLE_SIZE_BY_SEQ_LENGTH
    max_seq_length = MAX_SEQ_LENGTH  # also a parameter for the model
    vocab_size = VOCAB_SIZE  # also be a parameter for the model
    random_state = RANDOM_STATE

    dataset = CustomDataSet(
        sample_size_by_seq_length,
        max_seq_length,
        vocab_size,
        random_state,
    )

    train_data, test_data = train_test_split(dataset, test_size=0.2)

    batch_size = 8
    train_data_loader = DataLoader(train_data, batch_size, shuffle=True)

    test_data_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)

    iterator = iter(test_data_loader)
    input_test, target_test = next(iterator)  # one iteration only because batch size == size of dataset

    best_params = model_tuning(
        train_data_loader,
        input_test,
        target_test,
        vocab_size,
        max_seq_length,
        epochs=10,
        n_trials=20,
    )
    print("Best params: ", best_params)


def test_eq():
    a = torch.tensor([[1, 2], [3, 4], [5, 6]])
    b = torch.tensor([[1, 2], [3, 5], [5, 6]])

    c = torch.eq(a, b)
    print("torch.eq(a, b)")
    print(c)

    print("torch.all(c, dim=-1)")
    print(torch.all(c, dim=-1))


if __name__ == "__main__":
    # test_argmax()
    # test_mask()
    # test_unsqueeze()
    # test_cross_entrophy_loss()
    # print(1e-5 == 0.00001)
    # test_cat()
    # print(test_layer_norm())
    # test_optuna()
    test_eq()
