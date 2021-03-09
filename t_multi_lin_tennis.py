# TENNIS HUMIDITY PREDICTION
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

################################################## ÖN İŞLEME
veri = pd.read_csv('./src/odev_tenis.csv')

humidity = veri.loc[:,['humidity']]
veri.drop(['humidity'],inplace=True,axis=1)
veri = pd.concat([veri,humidity],axis=1)

veri = veri.replace({'windy': {False:0,True:1}})
veri = veri.replace({'play': {'no':0,'yes':1}})

print(veri)

outlook = veri.iloc[:,0].values

le = LabelEncoder()
outlook = le.fit_transform(outlook)
outlook = outlook.reshape((outlook.size,1))
ohe = OneHotEncoder()
onehotlabels = ohe.fit_transform(outlook).toarray()
veri.insert(0,'overcast',onehotlabels[:,0])
veri.insert(1,'rainy',onehotlabels[:,1])
veri.insert(2,'sunny',onehotlabels[:,2])
veri.drop(['outlook'],inplace=True,axis=1)

print(veri)
###############################################ELEMINATION

X_l = veri.iloc[:,[0,1,2,3,5]].values#Bagımsızlardan olusan columnları arraye donusturduk
Y_l = veri.loc[:,['humidity']].values#Bagımlı columnu arraye donusturduk

print(X_l)
print(Y_l)

r_ols = sm.OLS(endog = Y_l, exog = X_l,)#korelasyona baktık
r = r_ols.fit()#uyguladık
print(r.summary())#raporu aldık
################################################MODEL
bagimli = veri.loc[:,['humidity']]
bagimsiz = veri.drop(['humidity','windy'],inplace=False,axis=1)
x_train, x_test, y_train, y_test = train_test_split(bagimsiz,bagimli,test_size=0.33,random_state=0)# rastgele parçaladı

lr = LinearRegression()
lr.fit(x_train,y_train)
tahmin = lr.predict(x_test)

print(y_test)
print(tahmin)
print(veri)


