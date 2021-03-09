import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

veri = pd.read_csv('./src/veriler.csv')

x = veri.iloc[5:,1:4]#Cocuklar yok
y = veri.iloc[5:,-1:]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)# rastgele parçaladı

sc = StandardScaler()#Bazen daha iyi sonuclar verebiliyor standardize etmek
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

svm = SVC(kernel='sigmoid')# kernel = 'linear','poly','rbf','sigmoid' olabilir.
svm.fit(X_train,y_train)

y_pred = svm.predict(X_test)
y_pred = y_pred.reshape(6,1)
print(y_pred,type(y_pred))
print(y_test)


from sklearn.metrics import confusion_matrix
print('Başarı değerlendirme matrisi :')
print(confusion_matrix(y_test,svm.predict(X_test)))#Sol->sag diagonal dogru bilinen sayisi,diger diag yanlıs bilinen sayisi

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =[1,2,3,4,5,6,7,8,9,10]
y =[5,6,2,3,13,4,1,2,4,8]
z =[2,3,3,3,5,7,9,11,9,10]



ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()