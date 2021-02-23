import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from model_pl import RNN
from argparse import ArgumentParser
from dataset import NetflixDataset, collate_fn


def main(args):
    # ------------
    # model
    # ------------
    model = RNN(args)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    parser = RNN.add_model_specific_args(parser)

    ## DATA cli
    parser.add_argument(
        "--train_path", default="data/netflix/train_processed.csv", type=str
    )
    parser.add_argument(
        "--test_path", default="data/netflix/test_processed.csv", type=str
    )
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--n_workers", default=8, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--auto_select_gpus", default=True, type=bool)
    parser.add_argument("--log_gpu_memory", default=True, type=bool)
    parser.add_argument("--logs_path", default="runs/", type=str)

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

