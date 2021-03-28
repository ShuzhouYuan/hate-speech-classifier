#!/usr/bin/env python3
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

porter = PorterStemmer()

bad_words = []
with open('bad-words.txt') as f:
    for word in f:
        word = word.strip().lower()
        bad_words.append(word)
        bad_words.append(porter.stem(word))
bad_words = list(set(bad_words))

#print(bad_words)

df = pd.read_csv('./Davidson/test.csv')
tweets = df.tweet.values

bad = []
contain_bad = False
for t in tweets:
    words = word_tokenize(t)
    for w in words:
        word_origin = porter.stem(w.lower())
        if word_origin in bad_words:
            contain_bad = True
    if contain_bad:
        bad.append(1)
        contain_bad = False
    else:
        bad.append(0)

print(len(bad))
print(bad.count(0))
df['contain_bad'] = bad
df.to_csv('./Davidson/test-1.csv')