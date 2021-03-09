#ODEV2 -- Tahmin ve reg. sonu !
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

veri = pd.read_csv('./src/maaslar_yeni.csv')
print(veri)
print(veri.corr())
######################################################################################
#BACKWARD ELEMINATION--esik deger p-value = 0.05 den buykse alma
X_l = veri.iloc[:,[2]].values#Bagımsızlardan olusan columnları arraye donusturduk
Y_l = veri.loc[:,['maas']].values#Bagımlı columnu arraye donusturduk

print(X_l)
print(Y_l)

r_ols = sm.OLS(endog = Y_l, exog = X_l,)#korelasyona baktık
r = r_ols.fit()#uyguladık
print(r.summary())#raporu aldık
######################################################################################Unvan Seviyesi seçildi
x = veri.loc[:,['UnvanSeviyesi','Kidem','Puan']]
y = veri.iloc[:,-1:]
X = veri.loc[:,['UnvanSeviyesi','Kidem','Puan']].values
Y = veri.iloc[:,-1:].values
print(x)
print(y)
################################
#MLT
lr = LinearRegression()
lr.fit(x,y)

print('MLT-R2 score :',r2_score(Y,lr.predict(x)))
print('Specific Pre. :',lr.predict(np.array([[5,10,100]])))
################################
#Polly(PR)
poly_reg = PolynomialFeatures(degree=2)#Degree optimize değer için ayarlayabilirsin
x_poly = poly_reg.fit_transform(X)#Multiye benzetebilmek için fit_transforma soktuk

lr_poly = LinearRegression()
lr_poly.fit(x_poly,y)#olusturdugumuz x_poly üzerinden lin reg yaptık

print('Poly-R2 score:',r2_score(Y,lr_poly.predict(x_poly)))
x_poly = poly_reg.fit_transform(np.array([[5,10,100]]))
print('Specific Pre. :',lr_poly.predict(x_poly))
################################
#SVR
#Ölçeklendirme
sc = StandardScaler()#SVR için önce std olcekleme yapmak zorundayız
x_olcekli = sc.fit_transform(x)
y_olcekli = sc.fit_transform(y)
#SVR
svr = SVR(kernel='rbf')#radial basis func
svr.fit(x_olcekli,np.ravel(y_olcekli))#Model oluşturuldu
print('SVR-R2 score :',r2_score(y_olcekli,svr.predict(x_olcekli)))
################################
#DT
dec_tree = DecisionTreeRegressor(random_state=0)
dec_tree.fit(X,Y)
print('Decision tree-R2 score :',r2_score(Y,dec_tree.predict(X)))#Buna kanma cunku dec tree bu !
print('Specific Pre. :',dec_tree.predict(np.array([[5,10,100]])))
################################
#RR
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)#n_estimator kac tane dec tree cizilcegi yani datanın parcalancagı
rf_reg.fit(X,np.ravel(Y))
print('Random forest-R2 score:',r2_score(Y,rf_reg.predict(X)))

print('Specific Pre. :',rf_reg.predict(np.array([[5,10,100]])))



