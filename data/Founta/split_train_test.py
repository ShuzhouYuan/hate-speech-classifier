import pandas as pd
import numpy as np



df = pd.read_csv('./hatespeech_text_label_Founta.csv')

train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])
train.to_csv('./train.csv')
validate.to_csv('./validate.csv')
test.to_csv('./test.csv')