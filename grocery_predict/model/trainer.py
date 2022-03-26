import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from utils.preprocessor import Preprocessor

class GroceryTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = nn.LSTM(
            input_size = 12,
            hidden_size = 12,
            num_layers = 4
        )

        self.linear = nn.Linear(12, 1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-2)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        output = self.model(x)
        output = self.linear(output[0])
    
        loss = F.mse_loss(output[0], x)
        self.log('train_loss', loss)
        return loss

    def predict_step(self, test_batch, batch_idx):
        x, y = test_batch
        
        output = self.model(x)
        output = self.linear(output[0])
        output = torch.reshape(output, (6, 1))
        return output

class GroceryDataModule(pl.LightningDataModule):
    def __init__(self, batch_size) -> None:
        super(GroceryDataModule, self).__init__()
        self.batch_size = batch_size

    def setup(self, stage = None):
        ppro = Preprocessor(batch_size = self.batch_size)
        x_train, y_train, x_test, y_test, scaler, train_set_scaled, test_set_scaled, data_sales = ppro.preprocessor()

        if stage == "fit" :    
            self.x_train = x_train
            self.y_train = y_train
        elif stage == "predict":
            self.x_test = x_test
            self.y_test = y_test


    def train_dataloader(self):
        dataset = TensorDataset(self.x_train, self.y_train)
        sampler = RandomSampler(dataset)
        return DataLoader(
            dataset = dataset, 
            batch_size = self.batch_size, 
            sampler = sampler, 
            num_workers = 1
        )

    def predict_dataloader(self):
        dataset = TensorDataset(self.x_test, self.y_test)
        sampler = SequentialSampler(dataset)
        return DataLoader(
            dataset = dataset, 
            batch_size = self.batch_size, 
            sampler = sampler, 
            num_workers = 1
        )