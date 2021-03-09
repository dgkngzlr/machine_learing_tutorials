import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np

veri = pd.read_excel('./src/iris.xls')
print(veri)

x = veri.iloc[:,:4]
y = veri.iloc[:,-1:]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)# rastgele parçaladı


sc = StandardScaler()#Bazen daha iyi sonuclar verebiliyor standardize etmek
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
#Random forest
rfc = RandomForestClassifier(n_estimators=8, criterion='entropy')
rfc.fit(X_train,np.ravel(y_train))

tahmin = rfc.predict(X_test)
print(tahmin)

from sklearn.metrics import confusion_matrix
print('Başarı değerlendirme matrisi Random Forest:')
print(confusion_matrix(y_test,rfc.predict(X_test)))#Sol->sag diagonal dogru bilinen sayisi,diger diag yanlıs bilinen sayisi

#SVM

svm = SVC(kernel='linear')# kernel = 'linear','poly','rbf','sigmoid' olabilir.
svm.fit(X_train,np.ravel(y_train))

y_pred = svm.predict(X_test)


print('Başarı değerlendirme matrisi SVM :')
print(confusion_matrix(y_test,svm.predict(X_test)))#Sol->sag diagonal dogru bilinen sayisi,diger diag yanlıs bilinen sayisi
#KNN
knn = KNeighborsClassifier(n_neighbors=3,metric='minkowski')
knn.fit(X_train,np.ravel(y_train))

tahmin = knn.predict(X_test)


print('Başarı değerlendirme matrisi KNN:')
print(confusion_matrix(y_test,knn.predict(X_test)))#Sol->sag diagonal dogru bilinen sayisi,diger diag yanlıs bilinen sayisi
