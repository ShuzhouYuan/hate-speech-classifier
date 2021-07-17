import pandas as pd
import numpy as np



df = pd.read_csv('./arabic.csv')

train, test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df))])
train.to_csv('./train.csv')
test.to_csv('./test.csv')