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
words_tag = nltk.pos_tag(bad_words)
word_tag={word:tag for word,tag in words_tag}
tags = set(word_tag.values())
tags = ['<SWEAR-' + tag + '>' for tag in tags]
print(tags)

class BadWordsRemover:
    def __init__(self, file_path, bad_words):
        self.dataframe, self.tweets = self.read_csv(file_path)
        self.bad_words = bad_words

    @staticmethod
    def read_csv(file_path):
        df = pd.read_csv(file_path)
        return df, df.tweet.values

    def replace_bad_words(self, TOK):
        #contain_bad = []
        new_tweets = []
        for tweet in self.tweets:
            words = word_tokenize(tweet)
            for i in range(len(words)):
                word_origin = porter.stem(words[i].lower())
                if word_origin in self.bad_words:
                    words[i] = '<SWEAR-' + word_tag[word_origin] + '>'
            #contain_bad.append(1) if TOK in words else contain_bad.append(0)
            words = ' '.join(words)
            new_tweets.append(words)
        return new_tweets#, contain_bad

    def save_new_csv(self, TOK, filename):
        new_tweets = self.replace_bad_words(TOK)
        self.dataframe['tweet'] = new_tweets
        #self.dataframe['contain_bad'] = contain_bad
        self.dataframe.to_csv(filename)


if __name__ == '__main__':
    '''
    train_s = BadWordsRemover('../data/train.csv', bad_words)
    train_s.save_new_csv('<SWEAR>', 'train_<SWEAR>.csv')

    test_s =BadWordsRemover('../data/test.csv', bad_words)
    test_s.save_new_csv('<SWEAR>', 'test_<SWEAR>.csv')

    train_blank = BadWordsRemover('../data/train.csv', bad_words)
    train_blank.save_new_csv(' ', 'train_blank.csv')

    test_blank = BadWordsRemover('../data/test.csv', bad_words)
    test_blank.save_new_csv(' ', 'test_blank.csv')
    '''
    train_tag = BadWordsRemover('../data/train.csv', bad_words)
    train_tag.save_new_csv('<SWEAR>', 'train_tag.csv')

    test_tag = BadWordsRemover('../data/test.csv', bad_words)
    test_tag.save_new_csv('<SWEAR>', 'test_tag.csv')
