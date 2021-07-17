import pandas as pd

tweet, label = [], []
with open('test.txt', 'r') as f:
    for line in f:
        text, label1, label2 = line.split('\t')
        tweet.append(text)
        print(label2)
        if label2 == 'ABUSE\n':
            label.append(0)
        elif label2 == 'OTHER\n':
            label.append(2)
        elif label2 == 'INSULT\n' or 'PROFANITY\n':
            label.append(1)

print(label.count(0))
print(label.count(1))
print(label.count(2))
d={"text":tweet, "label":label}
df = pd.DataFrame(d)
df = df.sample(frac=1)
df.to_csv('test.csv')
