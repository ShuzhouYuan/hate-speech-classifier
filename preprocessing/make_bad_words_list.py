#!/usr/bin/env python3
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk


nltk.download('averaged_perceptron_tagger')

porter = PorterStemmer()

bad_words = []
with open('../data/bad-words.txt') as f:
    for word in f:
        word = word.strip().lower()
        bad_words.append(word)
        bad_words.append(porter.stem(word))
bad_words = list(set(bad_words))



df = pd.read_csv('../data/labeled_data.csv')
hate_content = df.loc[df['label'] == 0].tweet.values
hate_content = ' '.join(hate_content)

hate_content_list = word_tokenize(hate_content)
hate_content_list = list(set(hate_content_list))
hate_content_list_stem = [porter.stem(word) for word in hate_content_list] + hate_content_list

new_bad_words = []
for word in bad_words:
    if word in hate_content_list_stem:
        new_bad_words.append(word)
        new_bad_words.append(porter.stem(word))
new_bad_words = list(set(new_bad_words))
print(new_bad_words)

words_tag = nltk.pos_tag(new_bad_words)
word_tag = {word: tag for word, tag in words_tag}
tags = set(word_tag.values())
tags = ['<SWEAR-' + tag + '>' for tag in tags]
print(word_tag)

with open ('bad_words_in_hate.txt','a') as file:
    for word in new_bad_words:
        file.write(word+'\n')
