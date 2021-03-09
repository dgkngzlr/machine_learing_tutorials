import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np

veri = pd.read_csv('./src/veriler.csv')
x = veri.iloc[:,1:4]#Cocuklar var
y = veri.iloc[:,-1:]
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)# rastgele parçaladı

sc = StandardScaler()#Bazen daha iyi sonuclar verebiliyor standardize etmek
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

print(x_train)
print(y_train)

dct = DecisionTreeClassifier(criterion='entropy')#'gini' algoritmasıda olabilir.
dct.fit(X_train,y_train)
tahmin = dct.predict(X_test)

from sklearn.metrics import confusion_matrix
print('Başarı değerlendirme matrisi :')
print(confusion_matrix(y_test,dct.predict(X_test)))#Sol->sag diagonal dogru bilinen sayisi,diger diag yanlıs bilinen sayisi

