from asyncio import base_tasks
import sys

import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.lstm_sentiment import LSTMSentiment

class ClassificationTrainer(pl.LightningModule):
    def __init__(self, 
                 num_layers, 
                 vocab_size, 
                 hidden_dim, 
                 embedding_dim, 
                 lr,
                 output_dim = 1, 
                 dropout = 0.3,
                 batch_size = 50 ) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.batch_size = batch_size
        
        self.lr = lr

        self.model = LSTMSentiment(
            num_layers = num_layers,
            vocab_size = vocab_size,
            hidden_dim = hidden_dim,
            embedding_dim = embedding_dim,
            lr = lr,
            output_dim = output_dim,
            dropout = dropout
        )

        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim))
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim))

        h0 = h0.to(torch.device("cuda"))
        c0 = c0.to(torch.device("cuda"))

        self.hidden = (h0, c0)

        self.criterion = nn.BCELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def calc_accuracy(self, pred, y):
        pred = torch.round(pred.squeeze())
        return torch.sum(pred == y.squeeze()).item() / self.batch_size
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        h = tuple([each.data for each in self.hidden])

        out, hidden = self.model(x, h)
        loss = self.criterion(out.squeeze(), y.float())
        accuracy = self.calc_accuracy(out, y)

        self.log("accuracy", accuracy, prog_bar = True)
        return loss


    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch

        h = tuple([each.data for each in self.hidden])

        out, hidden = self.model(x, h)
        loss = self.criterion(out.squeeze(), y.float())
        accuracy = self.calc_accuracy(out, y)
        self.log("accuracy", accuracy, prog_bar = True)
        return loss