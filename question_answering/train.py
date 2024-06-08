import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)
from torch.optim import AdamW
from src.utils.train_utils import train
from src.optimizers.adasub import Adasub
from src.optimizers.adahessian import Adahessian

OPTIMIZERS = ["adamw", "adasub", "adahessian"]

def main():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing the data')
    parser.add_argument('--model_name', type=str, default='albert-base-v2', help=f'Name of the model')
    parser.add_argument('--tokenizer_name', type=str, default=None, help=f'Name of the tokenizer')
    parser.add_argument('--optimizer', type=str, choices=OPTIMIZERS, default="adahessian", help='Optimizer for training')
    parser.add_argument('--n_directions', type=int, default=2, help='The dimension of the subspace for Adasub')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs for training')
    parser.add_argument('--warmup_percent', type=float, default=0.1, help='Percentage of total training steps for warmup')
    args = parser.parse_args()

    data_dir = args.data_dir
    model_name = args.model_name
    tokenizer_name = model_name if not args.tokenizer_name else args.tokenizer_name
    optimizer_name = args.optimizer
    n_directions = args.n_directions
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    warmup_percent = args.warmup_percent

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    model.to(device)

    if optimizer_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, device=device)
    elif optimizer_name == "adasub":
        optimizer = Adasub(model.parameters(), lr=lr, n_directions=n_directions, device=device)
    elif optimizer_name == "adahessian":
        optimizer = Adahessian(model.parameters(), lr=lr, device=device)

    print("Start training...")
    model = train(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        data_dir = data_dir,
        batch_size=batch_size,
        device=device,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        warmup_percent=warmup_percent,
    )


if __name__ == "__main__":
    main()