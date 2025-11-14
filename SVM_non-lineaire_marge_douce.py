# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:31:47 2024

@author: charbons
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# Local library
import pandas as pd
# import conversion_data_base as conversion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import balanced_accuracy_score



# La librairie Panda permet de gérer de nombreux formats de données
import pandas as pd

# On charge les données
#X = pd.read_excel("data.xlsx",sheet_name=0,header=0,index_col=0)

plt.close('all')


iris = datasets.load_iris()

X = iris.data
yt = iris.target
target_names = iris.target_names
Xd=pd.DataFrame(data=X,columns=iris.feature_names)
Xs=X[:,2:4]
yt[yt==2]=0
plt.scatter(Xs[:,0],Xs[:,1],c=yt,marker='o')
#%%
# Apprentissage de la règle de décision
contrainte=1
svc = SVC( C=contrainte, kernel='rbf')
svc.fit(Xs,yt)

# Vecteurs supports obtenus après l'appentissage
sv=svc.support_vectors_
#Positions des vecteurs supports
print('position des points supports :',svc.support_vectors_)
print('nombre de points supports :',len(svc.support_vectors_))

#Coefficients des vecteurs supports
print(svc.dual_coef_)

plt.scatter(svc.support_vectors_[:,0],svc.support_vectors_[:,1],c='r', marker='*')

#%%
#Tracé de la frontière
x=np.arange(np.min(Xs[:,0]),np.max(Xs[:,0]),0.1)
y=np.arange(np.min(Xs[:,1]),np.max(Xs[:,1]),0.1)
trace=[]
exemple=np.zeros((2,1))
for i in x:
    for j in y:
        #exemple=[x[i] y[j]]
        exemple=np.array([i, j])
        classe=svc.predict(np.reshape(exemple,(1,2)))
        trace.append([exemple[0], exemple[1],classe[0]])
        
trace=np.asarray(trace)
plt.figure()
plt.scatter(trace[:,0],trace[:,1],c=trace[:,2],marker='.')
plt.scatter(Xs[:,0],Xs[:,1],c=yt,marker='o')
plt.show()