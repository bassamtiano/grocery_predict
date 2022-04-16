import os
import torch
from json.tool import main
from datasets import load_dataset
# dataset = load_dataset('reuters21578', 'ModHayes')
# dataset = load_dataset('imdb', 'plain_text')

import pytorch_lightning as pl

from utils.preprocessor import preprocessor

from utils.datamodule import ClassificationDataModule
from utils.trainer import ClassificationTrainer


if __name__ == "__main__":

    batch_size = 50

    pre = preprocessor(batch_size = batch_size)
    train_loader, val_loader, vocab_size = pre.load_dataset()

    device = torch.device("cuda")

    model = ClassificationTrainer(
        num_layers = 2,
        vocab_size = vocab_size + 1,
        hidden_dim = 256,
        embedding_dim = 64,
        lr = 1e-2
    )

    # dm = ClassificationDataModule (
    #     train_loader = train_loader,
    #     val_loader = val_loader
    # )

    

    trainer = pl.Trainer(gpus = 1, max_epochs = 10, default_root_dir = "./checkpoints/imdb")
    trainer.fit(model, train_loader, val_loader)



    # for row in dataset['train']:
    #     print(row.keys())
    #     print(row['topics'])
    #     break

    # print(len(dataset))

    # print(dataset.keys())
    # print(len(dataset["train"]))
    # print(len(dataset["test"]))

    # print(dataset["train"][0])
    # print(dataset["test"][0])

    