import pandas as pd


df = pd.read_csv('./german_dataset.tsv', sep="\t",error_bad_lines=False)
print(df.columns)

labels = df.task_2.values.tolist()
all_labels = set(labels)
print(all_labels)

print(labels.count('HATE'))
print(labels.count('NONE'))
print(labels.count('PRFN'))
print(labels.count('OFFN'))
num_label = []
for l in labels:
    if l == 'HATE':
        num_label.append(0)
    elif l == 'OFFN':
        num_label.append(1)
    elif l == 'NONE':
        num_label.append(2)
    elif l == 'PRFN':
        num_label.append(1)
print(len(num_label))
df['label'] = num_label
print(df)
#df.to_csv('./train_3_class.csv')
