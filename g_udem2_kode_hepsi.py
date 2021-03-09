import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

veri = pd.read_csv('eksikveriler.csv')

yas = veri[['yas']]
print(yas)
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')#nesne olustu

imputer = imputer.fit(yas)
yas = imputer.transform(yas)#NaN yerine ortalama koydu
print(yas[:,0])
yas_list = list(yas[:,0])
for i in range(len(veri['yas'].values)):#Değişikliği veride de gormek icin tum satır değiştirildi.
    veri['yas'].values[i] = yas_list[i]


print(veri)#Burada Nan lardan kurtulduk.

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
#veri.drop('ulke',inplace=True,axis=1)
print(veri)
veri.insert(0,'tr',onehotlabels[:,1])
veri.insert(1,'us',onehotlabels[:,2])
veri.insert(2,'fr',onehotlabels[:,0])

print(veri)#tr-us-fr
