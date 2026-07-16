"""
Build the paratope dataset
"""
import torch
from torch.utils.data import Dataset, DataLoader
from data_loader import list_paratope_files
from dataset_reader import read_parquet_files
from utils.tokenizer import build_dataset


class ParatopeDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        return {k: torch.tensor(v) for k, v in item.items()}


def get_split_path(split_name):
    files = list_paratope_files()
    return next(f for f in files if split_name in f.name)


def load_split_df(split_name):
    path = get_split_path(split_name)
    return read_parquet_files(path)


def build_split_dataset(tokenizer, split_name, max_length):
    df = load_split_df(split_name)
    encodings = build_dataset(tokenizer, df, max_length)
    return ParatopeDataset(encodings)


def build_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)