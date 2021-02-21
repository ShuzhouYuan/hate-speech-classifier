from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
import torch.nn as nn
import pickle
from lstm import LSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

batch_size = 32

with open('encoded_data', "rb") as file:
  train_dataset, test_dataset = pickle.load(file)

train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
test_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = batch_size)

lstm = LSTM(tokenizer.vocab_size, 300, 300, 3)
lstm = lstm.to(DEVICE)
print('from imported model')

optimizer = torch.optim.AdamW(lstm.parameters(), lr = 0.001)
loss_function = nn.CrossEntropyLoss()

epochs = 10

best_acc = 0
for e in range(epochs):
  print('training {} epoch...'.format(e+1))
  start_time = time.time()

  train_loss = 0

  lstm.train(True)

  for batch in train_dataloader:
    input = batch[0].to(DEVICE)
    label = batch[2].to(DEVICE)

    lstm.zero_grad()

    output = lstm(input)

    loss = loss_function(output, label)

    train_loss += loss.item()
    loss.backward()

    optimizer.step()

  num_total, num_correct = 0, 0
  lstm.train(False)
  with torch.no_grad():
    eval_loss = 0

    for batch in test_dataloader:
      lstm.zero_grad()

      input = batch[0].to(DEVICE)
      label = batch[2].to(DEVICE)

      output = lstm(input)
      loss = loss_function(output, label)

      predict_label = torch.argmax(output, dim=1)

      num_correct += (predict_label == label).sum().item()
      num_total += len(label)

      eval_loss += loss.item()

    acc = num_correct/num_total
    sec = time.time()-start_time
    if acc > best_acc:
      best_acc = acc
      torch.save(lstm, 'lstm.rnn')
  
  print('train_loss: {}, eval_loss: {}, accuracy: {}'.format(train_loss,eval_loss,acc))
  print('{} seconds used......'.format(sec))
