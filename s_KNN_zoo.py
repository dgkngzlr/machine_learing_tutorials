import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

veri = pd.read_csv('./src/zoo.csv')

x = veri.iloc[:,1:17]
y = veri.iloc[:,-1:]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)# rastgele parçaladı

sc = StandardScaler()#Bazen daha iyi sonuclar verebiliyor standardize etmek
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=3,metric='minkowski')
knn.fit(X_train,np.ravel(y_train))
ornek = np.array([[0,0,1,0,0,1,1,1,1,1,0,0,4,0,0,0]])
ornek = sc.transform(ornek)
print('Tahmin Listesi :')
print(knn.predict(X_test))
print('Gercek sınıflar:')
print(y_test)
print('Score :',knn.score(X_train,y_train))
print('Spesifik tahmin :',knn.predict(ornek))

from sklearn.metrics import confusion_matrix
print('Başarı değerlendirme matrisi :')
print(confusion_matrix(y_test,knn.predict(X_test)))#Sol->sag diagonal dogru bilinen sayisi,diger diag yanlıs bilinen sayisi





