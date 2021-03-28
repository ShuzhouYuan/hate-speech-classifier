import pandas as pd


df = pd.read_csv('./hatespeech_text_label_Founta.csv', error_bad_lines=False)
labels = df.label.values.tolist()
all_labels = set(labels)
print(all_labels)
print(labels.count('spam'))
print(labels.count('hateful'))
print(labels.count('abusive'))
print(labels.count('normal'))
num_label = []
for l in labels:
    if l == 'hateful':
        num_label.append(0)
    elif l == 'abusive':
        num_label.append(1)
    elif l == 'spam':
        num_label.append(2)
    elif l == 'normal':
        num_label.append(3)
print(len(num_label))
df['num_label'] = num_label
print(df)
df.to_csv('./hatespeech_text_label_Founta_1.csv')
