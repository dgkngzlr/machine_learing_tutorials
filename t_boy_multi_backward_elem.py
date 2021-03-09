import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

veri = pd.read_csv('./src/tip_don_veri.csv')
veri = veri.replace({'cinsiyet': {'e':1,'k':0}})#Cinsiyetteki e leri 1 k leri 0 yap.

bagimli = veri.loc[:,['boy']]
bagimsiz = veri.drop(['boy'],inplace=False,axis=1)

veri = pd.concat([bagimsiz,bagimli], axis = 1)
print(veri)

######################################################################################
#BACKWARD ELEMINATION--esik deger p-value = 0.05 den buykse alma
X_l = veri.iloc[:,[0,1,2,3,5]].values#Bagımsızlardan olusan columnları arraye donusturduk
Y_l = veri.loc[:,['boy']].values#Bagımlı columnu arraye donusturduk

print(X_l)
print(Y_l)

r_ols = sm.OLS(endog = Y_l, exog = X_l,)#korelasyona baktık
r = r_ols.fit()#uyguladık
print(r.summary())#raporu aldık

