import numpy as np
import pandas as pd
from difflib import SequenceMatcher
import re
import pickle
import time
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

#This script is used for plotting all the colours of one class

#read the data
ds = pd.read_pickle('dataset_full_5nn_names.pickle')
print(ds['Class'].value_counts(dropna=False))
#class to plot
colour = 'Green'

print('start plotting all',colour,'colours')
df = (ds.loc[ds['Class'] == colour]).reset_index()
col = pd.DataFrame({'R': (df['R']/255), 'G' : (df['G']/255), 'B' : (df['B']/255)})

data = ([(col.loc[0,'R'],col.loc[0,'G'],col.loc[0,'B'])])
tt = ([(col.loc[0,'R'],col.loc[0,'G'],col.loc[0,'B'])])
for j in range(0,len(col)):
    tt = ([(col.loc[j, 'R'], col.loc[j, 'G'], col.loc[j, 'B'])])
    for i in range(1,len(col)):
        dd = ((col.loc[j,'R'],col.loc[j,'G'],col.loc[j,'B']))
        tt.append(dd)
    if j == 0:
        data = [tt]
    else:
        data.append(tt)
plt.imshow(data)
plt.show()
