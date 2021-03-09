#Udemy 2.10 tr-us-fr numerik değere donusturme.ONEMLI
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

veri = pd.read_csv('veriler.csv')
##############################################Bu kodun tamamı çevirme islemini yapıyor.Onehot 2B matris ister
ulke = veri.iloc[:,0].values

le = LabelEncoder()
ulke = le.fit_transform(ulke)
print(type(ulke))
ulke = ulke.reshape((ulke.size,1))
print(ulke)

ohe = OneHotEncoder()
onehotlabels = ohe.fit_transform(ulke).toarray()
print(onehotlabels)
##############################################
veri.drop('ulke',inplace=True,axis=1)
print(veri)
veri.insert(0,'tr',onehotlabels[:,1])
veri.insert(1,'us',onehotlabels[:,2])
veri.insert(2,'fr',onehotlabels[:,0])
print(veri)
veri.to_csv('tip_don_veri.csv',index=False)