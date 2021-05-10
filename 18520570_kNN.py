import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix as pltcm 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = keras.datasets.mnist
 
# loading MNIST dataset
# verify
# the split between train and test is 60,000, and 10,000 respectly 
# one-hot is automatically applied


(X_trainFull, Y_trainFull), (X_testFull, Y_testFull) = data.load_data()
print(X_trainFull.shape[0], 'train samples')
print(X_testFull.shape[0], 'test samples')

X_short_train = X_trainFull[:600,:]
X_short_test = X_testFull[:100,:]
Y_short_train = Y_trainFull[:600]
Y_short_test = Y_testFull[:100]

pd.Series(Y_trainFull).value_counts()
pd.Series(Y_testFull).value_counts()

stat_train=dict()
for i in Y_trainFull:
    if i in stat_train:
        stat_train[i]+=1
    else:
        stat_train[i]=1

stat_test=dict()
for i in Y_testFull:
    if i in stat_test:
        stat_test[i]+=1
    else:
        stat_test[i]=1
t= X_short_train[0,:]
plt.imshow(t, cmap=plt.get_cmap('gist_gray'), interpolation='nearest')
plt.axis('off')

plt.figure(figsize=(60,60))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_short_train[i], cmap='gist_gray')
plt.show()

indices = np.where(Y_short_train==7)[0][:9]
plt.figure(figsize=(10,10))
for i,j in enumerate(indices):
    plt.subplot(3,3,i+1)
    digit = X_short_train[j]
    plt.imshow(digit, cmap = plt.cm.get_cmap('gist_gray'), interpolation = 'nearest')
    plt.axis('off')
plt.show()

X_train=X_short_train.reshape(600, 28*28)
X_test=X_short_test.reshape(100, 28*28)

for i in ['cosine', 'minkowski', 'euclidean']:
    print("Distance: ", i)
    for j in range(1,8,2):
      print("K=",str(j))
      classifier = KNeighborsClassifier(n_neighbors=j,metric=i)
      classifier.fit(X_train,Y_short_train)
      X_test=X_short_test.reshape(100,28*28)
      y_predict = classifier.predict(X_test)
      pltcm(classifier,X_test,Y_short_test)
      print("Acc = ",accuracy_score(Y_short_test, y_predict))

