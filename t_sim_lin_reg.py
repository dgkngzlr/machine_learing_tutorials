import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

veri = pd.read_csv('./src/satislar.csv')
print(veri)
x = veri.loc[:,'Aylar'].values
y = veri.loc[:,'Satislar'].values

x_train, x_test, y_train, y_test = train_test_split(veri[['Aylar']],veri[['Satislar']],
                                                    test_size=0.33,random_state=0)# rastgele parçaladı,içine df aldı

sc = StandardScaler()#Standardizasyon
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

print(X_train)
print(X_test)

# Model inşası (linear regression)
lr = LinearRegression()
lr.fit(x_train,y_train)#Standardize verilerde kullanılabilir
tahmin = lr.predict(x_test)
print('#################################')
print(tahmin)
print(x_test)
print(veri)
print('#################################')

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.title('Aylara Göre Satış')
plt.xlabel('Aylar')
plt.ylabel('Satışlar')
plt.plot(x_train,y_train,'r')#Eğtim verisi grafiği
plt.plot(x_test,lr.predict(x_test),'m')#Tahmin model grafiği
plt.scatter(x,y)#Tüm verinin grafiği scatter halde
plt.show()

