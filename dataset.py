import pandas as pd
import torch
import torch.utils.data as data
from torchvision import transforms
import ast
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    batch = [dataset[i] for i in N]
    """
    users = [item[0] for item in batch]
    pasts = [item[1] for item in batch]
    futures = [item[2] for item in batch]

    users = torch.LongTensor(users)
    pasts = pad_sequence(pasts)
    futures = torch.LongTensor(futures)
    return [users, pasts, futures]


class NetflixDataset(data.Dataset):
    """NetflixDataset dataset."""

    def __init__(
        self, csv_file,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.netflix_frame = pd.read_csv(
            csv_file,
            delimiter="\t",
            names=["user_id", "past", "future"],
            header=None,
            # iterator=True,
        )
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.netflix_frame)

    def __getitem__(self, idx):
        data = self.netflix_frame.loc[idx]
        user_id = data[0]
        try:
            past = torch.LongTensor(ast.literal_eval(data[1]))
        except:
            print(user_id)
            print(data[1])
            past = torch.LongTensor(ast.literal_eval(data[1]))
        future = data[2]
        return user_id, past, future

