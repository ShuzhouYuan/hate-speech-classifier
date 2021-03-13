import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertTokenizer, BertConfig, DistilBertTokenizerFast
from nltk.stem import PorterStemmer
import nltk
from sklearn.cluster import KMeans

config = BertConfig.from_pretrained('bert-base-cased', output_hidden_states=True)
tokenizer = DistilBertTokenizerFast.from_pretrained('bert-base-cased')


class Classifier(nn.Module):
  def __init__(self, num_label):
    super().__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased', config = config)
    self.linear = nn.Linear(768, num_label)

  def forward(self, input_ids, attention_mask):
    last_hidden_state, pooler_output, hidden_states = self.bert(input_ids, attention_mask, return_dict=False)
    logits = self.linear(pooler_output)
    return hidden_states, logits


model = torch.load('cls_model_with_raw_data', map_location=torch.device('cpu'))


def tokenize(text):
    """Given text, produces tokenization."""
    text = "[CLS] " + text + " [SEP]"
    text_tokens = tokenizer.tokenize(text)
    tokens = tokenizer.convert_tokens_to_ids(text_tokens)
    segs = [1] * len(tokens)
    return tokens, segs, dict(enumerate(text_tokens))

def encode(tokens, segs):
    """Returns hidden states for tokens"""
    tokens_tensor = torch.tensor([tokens])
    segments_tensor = torch.tensor([segs])
    model.eval()
    with torch.no_grad():
        hidden_states, output = model(tokens_tensor, segments_tensor)
    return hidden_states

def representation(enc):
    """Extracts representation for each token"""
    rep = torch.stack(enc).squeeze(1).permute(1, 0, 2)
    reps = []
    for token in rep:
        reps.append(token[0])  # maybe change this to 1
    return reps

def find_rep(txt):
    """Takes as input a string and returns the averaged representation of its tokens"""
    tokens, segs, t_idx = tokenize(txt)
    enc = encode(tokens, segs)
    rep = representation(enc)
    sent_rep = np.average([list(r) for r in rep], axis=0)
    return sent_rep.reshape(1, -1)


if __name__ == '__main__':

    porter = PorterStemmer()

    bad_words = []
    with open('../data/bad-words.txt') as f:
        for word in f:
            word = word.strip().lower()
            bad_words.append(word)
            bad_words.append(porter.stem(word))
    bad_words = list(set(bad_words))

    bad_words_rep = []

    for word in bad_words:
        rep = find_rep(word) # this is the uncontextualized rep of a single word, you can try it for all words
        bad_words_rep.append(rep)

    bad_words_rep = np.array(bad_words_rep).reshape((len(bad_words),-1))

    kmeans = KMeans(init = "random", n_clusters=3, n_init = 10, max_iter = 300,random_state = 42)
    kmeans.fit_transform(bad_words_rep)
    labels = kmeans.labels_

    print(labels)

    with open('bad_words_cluster_context.txt', 'a') as file:
        for word, label in zip(bad_words, labels):
            file.write(word+'\t'+str(label)+'\n')
