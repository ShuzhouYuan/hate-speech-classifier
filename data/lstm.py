import torch.nn as nn
import torch


class LSTM(nn.Module):
  def __init__(self,vocab_size, emb_size, lstm_size, num_label):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_size)
    self.lstm = nn.LSTM(emb_size, lstm_size, bidirectional=True, batch_first=True)
    self.linear = nn.Linear(lstm_size*2, num_label)

  def forward(self, input):
    emb =self.emb(input)
    lstm, _ = self.lstm(emb)
    lstm = torch.mean(lstm, dim=1)
    output = self.linear(lstm)
    return output