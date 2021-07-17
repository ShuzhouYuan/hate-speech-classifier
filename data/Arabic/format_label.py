import pandas as pd

tweet, label = [], []
with open('arabic.txt', 'r') as f:
    for line in f:
        label2 = line.split()[-1]
        text = line.split('\t')[0]
        tweet.append(text)
        print(label2)
        if label2 == 'hate':
            label.append(0)
        elif label2 == 'abusive':
            label.append(1)
        elif label2 == 'normal':
            label.append(2)

print(label.count(0))
print(label.count(1))
print(label.count(2))
print((len(label)))
print(len(tweet))
d={"text":tweet, "label":label}
df = pd.DataFrame(d)
#df = df.sample(frac=1)
#df.to_csv('arabic.csv')
