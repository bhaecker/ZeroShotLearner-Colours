import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d

#This script is used for platting all colours in RGB space

ds = pd.read_pickle('dataset_full_5nn_names.pickle')

g1 = (ds[ds['Class'] == 'Red']['R'],ds[ds['Class'] == 'Red']['G'],ds[ds['Class'] == 'Red']['B'])
g2 = (ds[ds['Class'] == 'Blue']['R'],ds[ds['Class'] == 'Blue']['G'],ds[ds['Class'] == 'Blue']['B'])
g3 = (ds[ds['Class'] == 'Green']['R'],ds[ds['Class'] == 'Green']['G'],ds[ds['Class'] == 'Green']['B'])
g4 = (ds[ds['Class'] == 'Yellow']['R'],ds[ds['Class'] == 'Yellow']['G'],ds[ds['Class'] == 'Yellow']['B'])
g5 = (ds[ds['Class'] == 'Cyan']['R'],ds[ds['Class'] == 'Cyan']['G'],ds[ds['Class'] == 'Cyan']['B'])
g6 = (ds[ds['Class'] == 'Magenta']['R'],ds[ds['Class'] == 'Magenta']['G'],ds[ds['Class'] == 'Magenta']['B'])
g7 = (ds[ds['Class'] == 'White']['R'],ds[ds['Class'] == 'White']['G'],ds[ds['Class'] == 'White']['B'])
g8 = (ds[ds['Class'] == 'Grey']['R'],ds[ds['Class'] == 'Grey']['G'],ds[ds['Class'] == 'Grey']['B'])
g9 = (ds[ds['Class'] == 'Black']['R'],ds[ds['Class'] == 'Black']['G'],ds[ds['Class'] == 'Black']['B'])

data = (g1,g2,g3,g4,g5,g6,g7,g8,g9)
colors = ("red","blue","green","yellow","cyan","magenta","white","grey","black")
groups= ("Red colours","Blue colours","Green colours","Yellow colours","Cyan colours","Magenta colours","White colours","Grey colours","Black colours")

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')

#fig = plt.figure()
#ax = fig.gca(projection='3d')

for data, color, group in zip(data, colors, groups):
    x, y, z = data
    ax.scatter(x, y, z, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

ax.set_xlabel('Red component')
ax.set_ylabel('Green component')
ax.set_zlabel('Blue component')
plt.title('3d colour classes')
plt.legend(loc=2)
#plt.show()

###############
#plot continous colours (no classification)
ax = fig.add_subplot(1, 2, 2, projection='3d')

#fig = plt.figure()
#ax = fig.gca(projection='3d')

for x, y, z in zip(ds['R'],ds['G'],ds['B']):
    ax.scatter(x, y, z, alpha=0.8, c=[(x/256,y/256,z/256)], edgecolors='none', s=30)

ax.set_xlabel('Red component')
ax.set_ylabel('Green component')
ax.set_zlabel('Blue component')
plt.title('3d continuous colour plot')
plt.legend(loc=2)
plt.show()


