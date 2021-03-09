#DECISION TREE PREDICTION---ASLINDA SINIFLANDIRMA MODELİ
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

veri = pd.read_csv('./src/maaslar.csv')
print(veri)

x = veri.iloc[:,1:2].values
y = veri.iloc[:,-1:].values

dec_tree = DecisionTreeRegressor(random_state=0)
dec_tree.fit(x,y)
print('Decision tree R2 degeri :',r2_score(y,dec_tree.predict(x)))#Buna kanma cunku dec tree bu !
plt.scatter(x,y,color='red')
plt.plot(x,dec_tree.predict(x))
print(dec_tree.predict(np.array([12]).reshape(-1,1)))
plt.show()


