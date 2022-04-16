import torch
import torch.nn as nn
import sys

class LSTMSentiment(nn.Module):
    def __init__(self,
                 num_layers, 
                 vocab_size, 
                 hidden_dim, 
                 embedding_dim, 
                 lr,
                 output_dim = 1, 
                 dropout = 0.3) -> None:
        super(LSTMSentiment, self).__init__()

        self.lr = lr

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = num_layers, batch_first = True)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, hidden):
        batch_size = x.size(0)
        embed = self.embedding(x)

        lstm_out, hidden = self.lstm(embed, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        out = out.view(batch_size, -1)
        out = out[:, -1]

        return out, hidden
