#RANDOM FOREST (ENSEMBLE METHOD)--ÇOKLU DECISION TREE
#df de array de verebilirsin
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

veri = pd.read_csv('./src/maaslar.csv')
print(veri)

x = veri.iloc[:,1:2]
y = veri.iloc[:,-1:].values.ravel()

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)#n_estimator kac tane dec tree cizilcegi yani datanın parcalancagı
rf_reg.fit(x,y)
print('Random forest R2 degeri :',r2_score(y,rf_reg.predict(x)))
plt.scatter(x,y,color='red')
plt.plot(x,rf_reg.predict(x))
plt.show()

print(rf_reg.predict(np.array([6.5]).reshape(-1,1)))#6.5 için değeri
print(np.array([6.5,5]).reshape(-1,1))