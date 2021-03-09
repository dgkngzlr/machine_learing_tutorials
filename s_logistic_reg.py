import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
veri = pd.read_csv('./src/veriler.csv')

x = veri.iloc[5:,1:4]#Cocuklar yok
y = veri.iloc[5:,-1:]
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)# rastgele parçaladı

print(x_train)
print(y_train)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)#Sadece transform cunku yukardaki x_train in ortalamasına gore transform etmesini istiyoruz

logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

tahmin = logr.predict(X_test)
print(x_test)
print(y_test)
print('Tahmmin :', tahmin)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,tahmin))#Sol->sag diagonal dogru bilinen sayisi,diger diag yanlıs bilinen sayisi
#Cocuklar marjinal oldukları için tahmin cok kotu.