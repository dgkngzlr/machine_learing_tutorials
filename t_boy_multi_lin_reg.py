import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

veri = pd.read_csv('./src/tip_don_veri.csv')
veri = veri.replace({'cinsiyet': {'e':1,'k':0}})#Cinsiyetteki e leri 1 k leri 0 yap.

bagimli = veri.loc[:,['boy']]
bagimsiz = veri.drop(['boy'],inplace=False,axis=1)
bagimsiz.drop(['yas','cinsiyet'],inplace=True,axis=1)#backward elem.
print(bagimsiz)

x_train, x_test, y_train, y_test = train_test_split(bagimsiz,bagimli,
                                                    test_size=0.33,random_state=0)# rastgele parçaladı

lr = LinearRegression()
lr.fit(x_train,y_train)
tahmin = lr.predict(x_test)

print(y_test)
print(tahmin)