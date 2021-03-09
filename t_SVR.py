import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

veri = pd.read_csv('./src/maaslar.csv')
print(veri)

x = veri.iloc[:,1:2]
y = veri.iloc[:,-1:]

x_std = np.std(veri.iloc[:,1:2].values)#Standart sapma
x_mean = np.mean(veri.iloc[:,1:2].values)#mean
y_std = np.std(veri.iloc[:,-1:].values)
y_mean = np.mean(veri.iloc[:,-1:].values)

sc = StandardScaler()#SVR için önce std olcekleme yapmak zorundayız
x_olcekli = sc.fit_transform(x)
y_olcekli = sc.fit_transform(y)

svr = SVR(kernel='rbf')#radial basis func
svr.fit(x_olcekli,np.ravel(y_olcekli))#Model oluşturuldu

tahmin_edilecek = (6.5-x_mean)/x_std#Tahmin edilcek degere standardizisyon uygulandı
tahmin_edilen = svr.predict(np.array([tahmin_edilecek]).reshape(-1,1))#Spesifik deger tahmini için input olarak STD deger aldı

tahmin = tahmin_edilen*y_std+y_mean#Tersine işlem yapılarak deger STD tipinden kurtuldu

print(tahmin_edilen)
print(tahmin)
print('SVR R2 degeri :',r2_score(y_olcekli,svr.predict(x_olcekli)))
plt.scatter(x_olcekli,y_olcekli)
plt.plot(x_olcekli,svr.predict(x_olcekli))
plt.show()