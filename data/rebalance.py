import pandas as pd

df = pd.read_csv('train.csv', names=['label','tweet'],header=0)
df = df.sort_values('label')
data_frame_0 = df.loc[df['label'] == 0]
data_frame_1 = df.loc[df['label'] == 1]
data_frame_2 = df.loc[df['label'] == 2]

num_0 = len(data_frame_0)
print(num_0)

data_frame_1 = data_frame_1.sample(frac=1)
data_frame_1 = data_frame_1[:num_0]

data_frame_2 = data_frame_2.sample(frac=1)
data_frame_2 = data_frame_2[:num_0]

new_df = [data_frame_0, data_frame_1, data_frame_2]
result = pd.concat(new_df)
result = result.sample(frac=1)
print((result))

result.to_csv('re_balance_train3.csv')
