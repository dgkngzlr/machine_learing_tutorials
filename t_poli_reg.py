import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.metrics import r2_score

veri = pd.read_csv('./src/maaslar.csv')
x = veri.loc[:,['Egitim Seviyesi']]
y = veri.iloc[:,-1:]


poly_reg = PolynomialFeatures(degree=4)#Degree optimize değer için ayarlayabilirsin
x_poly = poly_reg.fit_transform(x)#Multiye benzetebilmek için fit_transforma soktuk
print(x_poly)


lr = LinearRegression()
lr.fit(x_poly,y)#olusturdugumuz x_poly üzerinden lin reg yaptık

print('Poly R2 degeri :',r2_score(y,lr.predict(x_poly)))


print(lr.predict(poly_reg.fit_transform(np.array(6.5).reshape(1,-1))))#spesifik bir degerin tahmini degerini gormek içinde transform yapıyoruz

plt.scatter(x,y,color = 'red')
plt.plot(x,lr.predict(x_poly),color = 'blue')
plt.show()




#ornek = np.array([[11**0,11**1,11**2,11**3,11**4]])#4'e kadar cunku degree 4#spesifik bir deger

"""
veri = pd.read_csv('./src/maaslar.csv')
print(veri)
x = veri.loc[:,['Egitim Seviyesi']]
y = veri.iloc[:,-1:]
#Egitim test bolmesi yapılmıcak
lr = LinearRegression()
lr.fit(x,y)

plt.scatter(x,y,color = 'red')
plt.scatter(x,lr.predict(x),color = 'blue')
plt.show()
"""
