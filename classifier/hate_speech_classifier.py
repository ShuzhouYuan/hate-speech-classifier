from torchtext import data
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

data_csv = pd.read_csv('./labeled_data.csv', names=['label','tweet'],header=0)
#data_csv.loc[data_csv.label==0].sample(5)[['tweet', 'label']]

tweets = data_csv.tweet.values
labels = data_csv.label.values
print(len(labels))
print(len(tweets))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

max_length = 0
for t in tweets:
  ids = tokenizer.encode(t)
  max_length = max(len(ids),max_length)
print(max_length)

input_ids = []
attention_masks = []

for t in tweets:
  input_dict = tokenizer.encode_plus(t,add_special_tokens=True,max_length=max_length, truncation=True, padding='max_length',return_tensors='pt')
  input_ids.append(input_dict['input_ids'])
  attention_masks.append(input_dict['attention_mask'])
input_ids = torch.cat(input_ids,dim=0)
attention_masks = torch.cat(attention_masks,dim=0)

#print(tweets[0])
#print(input_ids[0])
#print(attention_masks[0])
#print(labels[0])
labels=torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 15

train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)

test_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = batch_size)

bert_model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 3)
bert_model = bert_model.to(DEVICE)

optimizer = AdamW(bert_model.parameters(),lr = 2e-5, eps = 1e-8)

epochs = 4

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

best_acc = 0
for e in range(epochs):
  print('training {} epoch...'.format(e+1))
  start_time = time.time()

  train_loss = 0

  bert_model.train(True)

  for batch in train_dataloader:
    input = batch[0].to(DEVICE)
    mask = batch[1].to(DEVICE)
    label = batch[2].to(DEVICE)

    bert_model.zero_grad()

    loss, output = bert_model(input_ids=input, attention_mask=mask, labels=label,return_dict=False)
    
    train_loss += loss.item()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
    
    optimizer.step()
    scheduler.step()

  num_total, num_correct = 0, 0
  bert_model.train(False)
  with torch.no_grad():
    eval_loss = 0

    for batch in test_dataloader:
      bert_model.zero_grad()

      input = batch[0].to(DEVICE)
      mask = batch[1].to(DEVICE)
      label = batch[2].to(DEVICE)

      loss, output = bert_model(input_ids=input, attention_mask=mask, labels=label,return_dict=False)

      predict_label = torch.argmax(output, dim=1)

      num_correct += (predict_label == label).sum().item()
      num_total += len(label)

      eval_loss += loss.item()

    acc = num_correct/num_total
    sec = time.time()-start_time
    if acc > best_acc:
      best_acc = acc
      torch.save(bert_model, 'model.bert')
  
  print('train_loss: {}, eval_loss: {}, accuracy: {}'.format(train_loss,eval_loss,acc))
  print('{} seconds used......'.format(sec))
