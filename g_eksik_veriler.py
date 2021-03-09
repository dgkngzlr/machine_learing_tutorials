# Udemy 2.9 eksik veriler.
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer#Nan değerleri mean ile değiştircek olan class improt edildi.

veri = pd.read_csv('./src/eksikveriler.csv')

# Sayısal veriler için Nan olan yere ortalama yazılabilir.
yas = veri[['yas']]
print(yas)
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')#nesne olustu

imputer = imputer.fit(yas)
yas = imputer.transform(yas)#NaN yerine ortalama koydu
print(yas[:,0])
yas_list = list(yas[:,0])

for i in range(len(veri['yas'].values)):#Değişikliği veride de gormek icin tum satır değiştirildi.
    veri['yas'].values[i] = yas_list[i]


print(veri)
