import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from Transformer import Transformer


def model_training(
    model: Transformer, train_data_loader, epochs: int, lr: int = 1e-5, print_loss: bool = False
) -> float:
    """
    Arguments
    ---------
    model: Transformer
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
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model.to(device)
    model.train()

    for epoch in range(epochs):
        for batch_input, batch_target in train_data_loader:
            optimizer.zero_grad()
            out = model(batch_input.to(device))
            loss = criterion(out.view(-1, model.tgt_vocab_size), batch_target.to(device).view(-1))
            loss.backward()
            optimizer.step()
        if print_loss:
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    return loss


def model_eval(model: Transformer, input_test: torch.Tensor, target_test: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    Arguments
    ---------
    model: Transformer
    input_test: torch.Tensor [sample_size, max_seq_length]
    target_test: torch.Tensor [sample_size, max_seq_length, tgt_vocab_size]

    Returns
    -------
    out_test: torch.Tensor [sample_size, max_seq_length, tgt_vocab_size]
    cross_entropy_loss: float
    """
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model.to(device)
    model.eval()

    with torch.no_grad():
        out_test = model(input_test.to(device))
        loss = criterion(out_test.view(-1, model.tgt_vocab_size), target_test.to(device).view(-1))

    cross_entropy_loss = loss.item()
    return out_test, cross_entropy_loss


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
    tgt_vocab_size = vocab_size + 1
    embed_dim = trial.suggest_categorical("embed_dim", [256, 512, 1024, 2048])
    max_seq_length = max_seq_length
    n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8])
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 4, 8])
    d_ff = trial.suggest_categorical("d_ff", [512, 1024, 2048, 4096])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    apply_mask = trial.suggest_categorical("apply_mask", [True, False])

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_dim=embed_dim,
        max_seq_length=max_seq_length,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
        apply_mask=apply_mask,
    )

    model_training(model, train_data_loader, epochs, lr=1e-5)
    cross_entropy_loss = model_eval(model, input_val, target_val)[1]
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


def print_accuracy(pred_test: torch.Tensor, target_test: torch.Tensor):
    """
    Report accuracy on token level and sequence level.

    Arguments
    ---------
    pred_test: torch.Tensor [test_sample_size, max_seq_length]
    target_test: torch.Tensor [test_sample_size, max_seq_length]
    """
    compare_tokens = torch.eq(pred_test.to("cpu"), target_test)

    n_true_predictions = torch.count_nonzero(compare_tokens)
    n_total_predictions = pred_test.size(0) * pred_test.size(1)
    print("Report accuracy on token level")
    print(f"Number of wrong token predictions: {n_total_predictions - n_true_predictions}")
    print(f"Number of total token predictions: {n_total_predictions}")
    print(f"Token Accuracy: {n_true_predictions / n_total_predictions * 100:.4f}%")

    n_true_predictions = torch.count_nonzero(torch.all(compare_tokens, dim=-1))
    n_total_predictions = pred_test.size(0)
    print("\nReport accuracy on sequence level")
    print(f"Number of wrong sequence predictions: {n_total_predictions - n_true_predictions}")
    print(f"Number of total sequence predictions: {n_total_predictions}")
    print(f"Sequence Accuracy: {n_true_predictions / n_total_predictions * 100:.4f}%")
