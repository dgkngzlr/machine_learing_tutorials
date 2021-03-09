# SAYISAL VERİLERİ MEAN VAL = 0 OLCAK ŞEKİLDE BELLİ BİR DÜZENE OTURTUR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


veri = pd.read_csv('./src/tip_don_veri.csv')

x = veri.drop('cinsiyet', inplace=False, axis=1)#Bağımsız değişkenler
y = veri[['cinsiyet']]#Bağımlı değişkenler

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)# rastgele parçaladı

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

print(X_train)
print(X_test)
