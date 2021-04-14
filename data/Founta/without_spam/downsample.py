import pandas as pd

df = pd.read_csv('./train.csv')
#df = df.sort_values('num_label')
data_frame_0 = df.loc[df['num_label'] == 0]
data_frame_1 = df.loc[df['num_label'] == 1]
data_frame_2 = df.loc[df['num_label'] == 2]


num_0 = len(data_frame_0)
num_1 = len(data_frame_1)
num_2 = len(data_frame_2)
print(num_0, num_1, num_2)

data_frame_1 = data_frame_1.sample(frac=1)
data_frame_1 = data_frame_1[:int(num_1/2)]

data_frame_2 = data_frame_2.sample(frac=1)
data_frame_2 = data_frame_2[:int(num_2/2)]

#data_frame_3 = data_frame_3.sample(frac=1)
#data_frame_3 = data_frame_3[:num_2]

#data_frame_3['num_label'] = 2

new_df = [data_frame_0, data_frame_1, data_frame_2]
result = pd.concat(new_df)
result = result.sample(frac=1)
print((result))

result.to_csv('./train_half.csv')
