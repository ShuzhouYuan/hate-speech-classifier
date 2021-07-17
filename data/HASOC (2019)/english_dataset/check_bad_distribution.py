import pandas as pd

df = pd.read_csv('./HASOC_contain_bad.csv')

labels = df.label.values
bads = df.contain_bad.values

a_x, a_y, b_x, b_y, c_x, c_y, d_x, d_y = 0,0,0,0,0,0,0,0

for label, bad in zip(labels, bads):
    if label == 0 and bad == 0:
        a_x+=1
    elif label == 0 and bad ==1:
        a_y+=1
    elif label == 1 and bad == 0:
        b_x+=1
    elif label == 1 and bad ==1:
        b_y+=1
    elif label == 2 and bad ==0:
        c_x+=1
    elif label ==2 and bad ==1:
        c_y +=1
    elif label == 3 and bad ==0:
        d_x+=1
    elif label == 3 and bad == 1:
        d_y+=1

bad_in_hate = a_y/(a_x+a_y)
bad_in_off = b_y/(b_x+b_y)
bad_in_spa = c_y/(c_x+c_y)
bad_in_nor = d_y/(d_x+d_y)


print(a_y, a_x+a_y, b_y, b_x+b_y, c_y, c_x+c_y, d_y, d_x+d_y)
print(bad_in_hate, bad_in_off, bad_in_spa, bad_in_nor)