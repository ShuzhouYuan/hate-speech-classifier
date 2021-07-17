import pandas as pd
import numpy as np



df = pd.read_csv('../train.csv')

train, validate = np.split(df.sample(frac=1, random_state=42), [int(.9*len(df))])
train.to_csv('./train.csv')
validate.to_csv('./dev.csv')
