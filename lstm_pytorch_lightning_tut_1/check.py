import pandas as pd
import torch

from models.lstm_sentiment import LSTMSentiment

# df = pd.read_csv("datasets/imdb/IMDB Dataset.csv")
# print(len(df["review"]))

def predict():
    data_checkpoint = torch.load("./checkpoints/imdb/lightning_logs/version_0/checkpoints/epoch=9-step=7499.ckpt")

    model = LSTMSentiment(
        num_layers = 2,
        vocab_size = 1000 + 1,
        hidden_dim = 256,
        embedding_dim = 64,
        lr = 1e-2,
        output_dim = 1,
        dropout = 0.3
    )

    model.load_state_dict(data_checkpoint["state_dict"])


if __name__ == "__main__":
    predict()
    