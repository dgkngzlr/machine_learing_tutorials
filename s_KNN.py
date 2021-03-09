import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
veri = pd.read_csv('./src/veriler.csv')

x = veri.iloc[5:,1:4]#Cocuklar yok
y = veri.iloc[5:,-1:]
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)# rastgele parçaladı

sc = StandardScaler()#Bazen daha iyi sonuclar verebiliyor standardize etmek
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

print(x_train)
print(y_train)

knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(X_train,np.ravel(y_train))
print(knn.score(X_train,y_train))

print(knn.predict(X_test))
print(y_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,knn.predict(X_test)))#Sol->sag diagonal dogru bilinen sayisi,diger diag yanlıs bilinen sayisi
