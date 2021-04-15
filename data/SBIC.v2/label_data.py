import pandas as pd
from collections import defaultdict


tweet_and_label = defaultdict(lambda: defaultdict(int))

df = pd.read_csv('SBIC.v2.tst.csv')

tweets = df.post.values
hate_labels = df.intentYN.values
off_labels = df.offensiveYN.values

for tweet, hate, off in zip(tweets,hate_labels,off_labels):
    if hate >= 0.5:
        tweet_and_label[tweet]['hate_Y'] += 1
    if hate < 0.5:
        tweet_and_label[tweet]['hate_N'] += 1
    if off >= 0.5:
        tweet_and_label[tweet]['off_Y'] += 1
    if off < 0.5:
        tweet_and_label[tweet]['off_N'] += 1

labels = []
for tweet in tweet_and_label.keys():
    hate_y, hate_n, off_y, off_n = tweet_and_label[tweet]['hate_Y'],\
                                   tweet_and_label[tweet]['hate_N'],\
                                   tweet_and_label[tweet]['off_Y'],\
                                   tweet_and_label[tweet]['off_N']
    if hate_y >= hate_n:
        labels.append(0)
    elif off_y >= off_n:
        labels.append(1)
    elif off_n > off_y:
        labels.append(2)


print(labels.count(0), labels.count(1), labels.count(2))

d = {'tweet': list(tweet_and_label.keys()), 'label': labels}

df = pd.DataFrame(d)
df = df.sample(frac=1)
df.to_csv('test.csv')
