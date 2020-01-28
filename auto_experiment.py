import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import pandas as pd
import pickle
from sklearn import datasets, linear_model, preprocessing, decomposition, manifold, neural_network, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


#This script is used to carry out the experiment, where we train a classifier on six classes and use the attribute matrix to classify the 3 unseen classes


#load data
ds_u = pd.read_pickle('dataset_full_5nn_names.pickle')
ds_b = pd.read_pickle('dataset_a23_balanced_seed12_5nn.pickle')
Smatrix = pd.read_pickle('Smatrix_full_5nn_without_names.pickle')

print(ds['Class'].value_counts())

#comb = list(combinations(['Green','Blue','Grey','White','Black','Red','Yellow','Magenta','Cyan'], 3))

comb = list(combinations(['Green','Blue','Grey','White'],3)) #for quick result
results_u = pd.DataFrame(columns=['Accuracy on seen classes','Accuracy on unseen classes'])
results_b = pd.DataFrame(columns=['Accuracy on seen classes','Accuracy on unseen classes'])

for team in zip((ds_u,ds_b),(results_u,results_b)):
    ds = team[0]
    results = team[1]
    for j in range(0,len(comb)):
        col0 = comb[j][0]
        col1 = comb[j][1]
        col2 = comb[j][2]
        #print(col0,col1,col2)

        # get data, which is available in training stage
        X = ds[(ds.Class != col0) & (ds.Class != col1) & (ds.Class != col2)][['R', 'G', 'B']].reset_index().drop(
            columns=['index'])
        Y = ds[(ds.Class != col0) & (ds.Class != col1) & (ds.Class != col2)][['Class']].values[:, 0]
        S = np.matrix(Smatrix.drop(columns=[col0, col1, col2]), dtype=float)

        ##Logistic regression
        model = linear_model.LogisticRegression(random_state=1)
        #Support Vector machines
        #model = LinearSVC(max_iter = 10000)
        model.fit(X, Y)

        results.loc[j, 'Accuracy on seen classes'] = np.mean(cross_val_score(model, X, Y, cv=5))

        # get W
        W = model.coef_.T
        # get V
        V = np.linalg.lstsq(S.T, W.T)[0].T

        # print(V.shape,Sprime.shape)

        # get data which is available after training
        Sprime = np.matrix(Smatrix[[col0, col1, col2]], dtype=float)
        # print(X.shape,Y.shape,S.shape,Sprime.shape)
        Xprime = ds[(ds.Class == col0) | (ds.Class == col1) | (ds.Class == col2)][['R', 'G', 'B']].reset_index().drop(
            columns=['index'])
        Yprime = ds[(ds.Class == col0) | (ds.Class == col1) | (ds.Class == col2)][['Class']].values[:, 0]

        # get W'
        Wprime = np.dot(V, Sprime)

        # make pred on unseen data
        Ypred = np.zeros_like(Yprime)
        for i in range(0, len(Yprime)):
            x = np.array([[Xprime.loc[i, 'R']], [Xprime.loc[i, 'G']], [Xprime.loc[i, 'B']]]).T
            id = np.argmax(np.dot(x, Wprime))
            if id == 0:
                Ypred[i] = col0
            if id == 1:
                Ypred[i] = col1
            if id == 2:
                Ypred[i] = col2

        #print(metrics.confusion_matrix(Yprime, Ypred, labels=[col0, col1, col2]))

        #print(accuracy_score(Yprime, Ypred))
        results.loc[j, 'Accuracy on unseen classes'] = accuracy_score(Yprime, Ypred)
        #print(classification_report(Yprime,Ypred))


results_u = results_u.astype(float)
results_b = results_b.astype(float)

print('stat for unbalanced classes:',results_u.describe(include = 'all'))
print('stat for balanced classes:',results_b.describe(include = 'all'))

print('unbalanced classes:')
print('Triple with MAX acc. on other six classes',comb[results_u['Accuracy on seen classes'].idxmax()],results_u['Accuracy on seen classes'].max())
print('Triple with MAX acc on unseen classes',comb[results_u['Accuracy on unseen classes'].idxmax()],results_u['Accuracy on unseen classes'].max())
print('Triple with MIN acc. on other six classes',comb[results_u['Accuracy on seen classes'].idxmin()],results_u['Accuracy on seen classes'].min())
print('Triple with MIN acc on unseen classes',comb[results_u['Accuracy on unseen classes'].idxmin()],results_u['Accuracy on unseen classes'].min())
print('balanced classes:')
print('Triple with MAX acc. on other six classes',comb[results_b['Accuracy on seen classes'].idxmax()],results_b['Accuracy on seen classes'].max())
print('Triple with MAX acc on unseen classes',comb[results_b['Accuracy on unseen classes'].idxmax()],results_b['Accuracy on unseen classes'].max())
print('Triple with MIN acc. on other six classes',comb[results_b['Accuracy on seen classes'].idxmin()],results_b['Accuracy on seen classes'].min())
print('Triple with MIN acc on unseen classes',comb[results_b['Accuracy on unseen classes'].idxmin()],results_b['Accuracy on unseen classes'].min())


plt.plot(results_u['Accuracy on seen classes'], results_u['Accuracy on unseen classes'], 'ro',mew=1,label='unbalanced classes')
plt.plot(results_b['Accuracy on seen classes'], results_b['Accuracy on unseen classes'], 'bo',mew=1,label='balanced classes')
plt.plot(results_u['Accuracy on seen classes'].mean(), results_u['Accuracy on unseen classes'].mean(), 'rx',markersize=12,label='mean of unbalanced classes')
plt.plot(results_b['Accuracy on seen classes'].mean(), results_b['Accuracy on unseen classes'].mean(), 'bx',markersize=12,label='mean of balanced classes')

plt.xlabel('Accuracy on seen classes')
plt.ylabel('Accuracy on unseen classes')
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0, 1], [0.33, 0.33], 'k-')
plt.show()