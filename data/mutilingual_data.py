import pandas as pd

df = pd.read_csv('./Arabic/new/test.csv')

arabic_labels = df.label.values.tolist()
arabic_text = df.text.values.tolist()

df = pd.read_csv('./Davidson/new/test.csv')

davidson_labels = df.label.values.tolist()
davidson_text = df.tweet.values.tolist()

df = pd.read_csv('./Founta/without_spam/new/test.csv')

founta_labels = df.num_label.values.tolist()
founta_text = df.tweet.values.tolist()

df = pd.read_csv('./GermEval/new/test.csv')

germeval_labels = df.label.values.tolist()
germeval_text = df.text.values.tolist()

df = pd.read_csv('./HASOC (2019)/english_dataset/new/test.csv')

hasoc_en_labels = df.label.values.tolist()
hasoc_en_text = df.text.values.tolist()

df = pd.read_csv('./HASOC (2019)/german_dataset/new/test.csv')

hasoc_de_labels = df.label.values.tolist()
hasoc_de_text = df.text.values.tolist()

df = pd.read_csv('./HASOC (2019)/hindi_dataset/new/test.csv')

hasoc_hin_labels = df.label.values.tolist()
hasoc_hin_text = df.text.values.tolist()

text = arabic_text + davidson_text + founta_text + germeval_text + hasoc_en_text + hasoc_de_text + hasoc_hin_text
label = arabic_labels + davidson_labels + founta_labels + germeval_labels + hasoc_en_labels + hasoc_de_labels + hasoc_hin_labels

d={"text":text, "label":label}
df = pd.DataFrame(d)
df = df.sample(frac=1)

labels = df.label.values.tolist()
print(labels.count(0), labels.count(1), labels.count(2), labels.count(3) )

#df.to_csv('./multilingual/new/train_multi.csv')
