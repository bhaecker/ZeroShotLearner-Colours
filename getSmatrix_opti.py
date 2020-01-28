import numpy as np
import pandas as pd
from difflib import SequenceMatcher
import re
import pickle
import time
import csv
from sklearn.neighbors import KNeighborsClassifier

#This script is used to produce the attribute matrix (Smatrix), which is needed for the learning transfer

# helper functions:

# define a function to unify a list
def unify(seq, idfun=None):
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result


# function for detecting a word in a string
def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search


#########################################################################################

# read the data
ds = pd.read_pickle('dataset_full_5nn_without_names.pickle')
print(ds['Class'].value_counts(dropna=False))

# get all the words in the names
words = ds['name'][0].split()
print(ds['name'])
for i in range(len(ds.index) - 1):
    words = words + (ds['name'][i + 1].split())

# unify 'words' (get rid of multiple occurrences of words) and take the result as attributes
attr = sorted(unify(words))

# remove words which have no semantic meaning
with open('unigram_freq.csv', newline='') as file:
    reader = csv.reader(file)
    nonsemanticwords = list(reader)

for i in nonsemanticwords:
    try:
        attr.remove(i[0])
    except ValueError:
        pass


##################################################################################################

###create Smatrix:
print('start')
S = pd.DataFrame(index=unify(attr), columns=unify(ds['Class']))
for colour in S.columns.format():
    start = time.time()
    df = ds.loc[ds['Class'] == colour, 'name'].reset_index(drop=True)
    words_in_class = df[0].split()
    for i in range(len(df.index) - 1):
        words_in_class = words_in_class + (df[i + 1].split())
    for word in S.index.format():
        S.loc[word, colour] = words_in_class.count(word)
    S[colour] = S[colour] / S[colour].max()
    print(colour)
    end = time.time()
    print(end - start)

print(S)

S.to_pickle('Smatrix_full_5nn_without_names.pickle')

for colour in S.columns:
    print(colour, 'has the count:')
    print(S[S[colour] != 0][colour].sort_values(ascending=False))
