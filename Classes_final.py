import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d

#This script is used for mapping each colour to a class

#read the data
ds = pd.read_csv('colournames.csv', sep=",",encoding='latin-1', header=None)
ds.columns = ["name","hex"]

#randomly delete rows, s.t. we are left with 'size' colours
size = 18181
np.random.seed(20)
remove_n = 18181 - size
drop_indices = np.random.choice(ds.index, remove_n, replace=False)
ds = ds.drop(drop_indices)
ds = ds.reset_index(drop=True)


#add columns with the coresponding RGB values
ds['R'], ds['G'], ds['B'] = np.nan,np.nan,np.nan
for j in range(len(ds.index)):
    h = ds['hex'][j].lstrip('#')
    trip = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    ds.loc[j,'R'], ds.loc[j,'G'], ds.loc[j,'B'] = trip[0], trip[1], trip[2]
#add classes
ds['Class'] = np.nan

##parameters determining the classes:
a=23

for i in range(len(ds.index)):
    if ds['R'][i] < a+32 and ds['B'][i] <a+32 and ds['G'][i] <a+32:
        ds.loc[i,'Class'] = 'Black'
    elif (ds['G'][i] - ds['R'][i]) > a-20 and (ds['G'][i] - ds['B'][i]) > a-20:
        ds.loc[i,'Class'] = 'Green'
    elif (ds['B'][i]-ds['R'][i]) > a and (ds['B'][i] - ds['G'][i]) > a:
        ds.loc[i,'Class'] = 'Blue'
    elif ds['R'][i] > (255-a) and ds['B'][i] >(255-a) and ds['G'][i] >(255-a):
        ds.loc[i,'Class'] = 'White'
    elif (ds['R'][i] - ds['G'][i]) > a+10 and (ds['R'][i] - ds['B'][i]) > a+10:
        ds.loc[i,'Class'] = 'Red'
    elif abs(ds['R'][i] - ds['B'][i]) < (a-10):
        ds.loc[i,'Class'] = 'Magenta'
    elif abs(ds['R'][i] - ds['G'][i]) < (a-5) and abs(ds['R'][i] - ds['B'][i]) < (a-5) and abs(ds['B'][i] - ds['G'][i]) < (a):
        ds.loc[i,'Class'] = 'Grey'
    elif abs(ds['R'][i] - ds['G'][i]) < (a-20):
        ds.loc[i,'Class'] = 'Yellow'
    elif abs(ds['B'][i] - ds['G'][i]) > (a+30): #and ds['R'][i] < a:
        ds.loc[i,'Class'] = 'Cyan'

print('after mapping',ds['Class'].value_counts(dropna=False),len(ds.index))

###find the classes in the unique colour names
classes = ['green','blue','red','grey','white','black','yellow','turquoise','cyan','magenta','purple','Green','Blue','Grey','White','Black','Red','Yellow','Magenta','Purple','Turquoise','Cyan']
for i in range(len(ds.index)):
    for colour in classes:
        #print(set(ds.loc[i, 'name'].split()),colour)
        if colour in set(ds.loc[i, 'name'].split()):
            ds.loc[i, 'Class'] = colour

ds.loc[ds['Class'] == 'red','Class'] = 'Red'
ds.loc[ds['Class'] == 'green','Class'] = 'Green'
ds.loc[ds['Class'] == 'blue','Class'] = 'Blue'
ds.loc[ds['Class'] == 'yellow','Class'] = 'Yellow'
ds.loc[ds['Class'] == 'white','Class'] = 'White'
ds.loc[ds['Class'] == 'black','Class'] = 'Black'
ds.loc[ds['Class'] == 'grey','Class'] = 'Grey'
ds.loc[ds['Class'] == 'cyan','Class'] = 'Cyan'
ds.loc[ds['Class'] == 'Turquoise','Class'] = 'Cyan'
ds.loc[ds['Class'] == 'turquoise','Class'] = 'Cyan'
ds.loc[ds['Class'] == 'magenta','Class'] = 'Magenta'
ds.loc[ds['Class'] == 'purple','Class'] = 'Magenta'
ds.loc[ds['Class'] == 'Purple','Class'] = 'Magenta'

########################################
print('after names mapping',ds['Class'].value_counts(dropna=False),len(ds.index))


classifier = KNeighborsClassifier(n_neighbors=5)
X_train = ds[ds['Class'].notnull()][['R','G','B']]
y_train = ds[ds['Class'].notnull()][['Class']]
X_empty =  ds[ds['Class'].isnull()][['R','G','B']]
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_empty)
j = 0
#print(X_empty.index)
for i in X_empty.index:
    ds.loc[i,'Class'] = y_pred[j]
    #print(ds[[R,G,B]],y_pred[j])
    j = j+1

print('after Knn',ds['Class'].value_counts(dropna=False))


###drop the instances with class name in it

classes = ['green','blue','red','grey','white','black','yellow','turquoise','cyan','magenta','purple','Green','Blue','Grey','White','Black','Red','Yellow','Magenta','Purple','Turquoise','Cyan']

for colour in classes:
    for i in range(len(ds.index)):
        #print(set(ds.loc[i, 'name'].split()),colour)
        if colour in set(ds.loc[i, 'name'].split()):
            ds = ds.drop([i])
    ds = ds.reset_index(drop=True)

ds = ds.reset_index()
print('after dropping names',ds['Class'].value_counts(dropna=False),len(ds.index))

colour = 'Yellow'

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

