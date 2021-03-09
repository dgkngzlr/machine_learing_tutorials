import pandas as pd
from sklearn.model_selection import train_test_split

veri = pd.read_csv('./src/tip_don_veri.csv')

x = veri.drop('cinsiyet', inplace=False, axis=1)#Bağımsız değişkenler
y = veri[['cinsiyet']]#Bağımlı değişkenler

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)# rastgele parçaladı

print(x_train)
print('##################')
print(x_test)


"""
    x_train = Eğitim verisinin bağımsız değişkenli kısmı
    y_train = Eğitim verisinin bağımlı değişkenli kısmı
    
    x_test = Test verisinin bağımsız değişkenli kısmı
    y_test = Test verisinin bağımlı değişkenli kısmı
"""
