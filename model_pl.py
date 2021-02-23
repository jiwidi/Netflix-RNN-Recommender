import torch.nn as nn
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from dataset import collate_fn, NetflixDataset


class RNN(pl.LightningModule):
    def __init__(
        self,
        args=None,
        embedding_dim=300,
        hidden_dim=512,
        n_layers=3,
        n_users=476422,
        n_movies=17771,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.args = args
        self.embeddings_user = nn.Embedding(
            self.hparams.n_users, self.hparams.embedding_dim
        )
        self.embeddings_past = nn.Embedding(
            self.hparams.n_movies, self.hparams.embedding_dim
        )

        self.lstm = nn.LSTM(
            self.hparams.embedding_dim,
            self.hparams.hidden_dim,
            num_layers=self.hparams.n_layers,
        )
        self.linear = nn.Linear(
            self.hparams.hidden_dim + self.hparams.embedding_dim, self.hparams.n_movies
        )

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        user, past = x
        user = self.embeddings_user(user)
        past = self.embeddings_past(past)
        lstm_out, self.hidden = self.lstm(past)

        concat = torch.cat((user, lstm_out[-1]), -1)
        return self.linear(concat)

    def training_step(self, batch, batch_idx):
        user, past, target = batch[0], batch[1], batch[2]
        out = self((user, past))
        loss = self.criterion(out, target)
        return loss

    def validation_step(self, batch, batch_idx):
        user, past, target = batch[0], batch[1], batch[2]
        out = self((user, past))
        loss = self.criterion(out, target)
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        user, past, target = batch[0], batch[1], batch[2]
        out = self((user, past))
        loss = self.criterion(out, target)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embedding_dim", type=int, default=300)
        parser.add_argument("--hidden_dim", type=int, default=512)
        parser.add_argument("--n_layers", type=int, default=3)
        parser.add_argument("--n_users", type=int, default=476422)
        parser.add_argument("--n_movies", type=int, default=17771)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        return parser

    ####################
    # DATA RELATED HOOKS
    ####################

    # def prepare_data(self):
    #     # download
    #     dataset = NetflixDataset(args.train_path)
    #     dataset = NetflixDataset(args.test_path)

    def setup(self, stage=None):
        print("Loading datasets")
        self.train_dataset = NetflixDataset(self.args.train_path)
        self.test_dataset = NetflixDataset(self.args.test_path)
        print("Done")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.n_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.n_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.n_workers,
            collate_fn=collate_fn,
        )
