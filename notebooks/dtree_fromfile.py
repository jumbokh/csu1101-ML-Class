#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv (r'students.csv')
print (df.head())
print(df.shape)

df['class'] = df['class'].map( {'A': 1, 'B': 2} ).astype(int)
df['Gendar'] = df['Gendar'].map( {'F': 1, 'M': 0} ).astype(int)
df['player'] = df['player'].map( {'y': 1, 'n': 0} ).astype(int)
print (df.head())
print(df.info())