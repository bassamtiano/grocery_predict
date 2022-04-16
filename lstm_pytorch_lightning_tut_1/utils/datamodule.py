import torch
from torch.utils.data import random_split, DataLoader, TensorDataset, RandomSampler

import pytorch_lightning as pl

from utils.preprocessor import preprocessor

from datasets import load_dataset

class ClassificationDataModule(pl.LightningDataModule) :
    def __init__(self, train_loader, val_loader) -> None:
        super(ClassificationDataModule).__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
