import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

veri = pd.read_csv('./src/gyro.csv')

veri = veri.iloc[:,1::2]

x = veri.iloc[:,:2]
y = veri.iloc[:,2:3]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

y_train = y_train.values.ravel()

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(x_train,y_train)
joblib.dump(rf_reg,'./rf_roll.joblib')
print('Random forest R2 degeri :',r2_score(y_test.values.ravel(),rf_reg.predict(x_test)))
print(rf_reg.predict(x_test))