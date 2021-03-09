#BAŞLANGIÇ SEVİYESİ#
import pandas as pd
import random

#İlk önce pandas serisi oluşturalım.İndex hariç tek sütundan oluşur.
pandas_serisi = pd.Series(data=['Dogukan', 11, 3.5, 'ESTU'], index=['İsim', 'Kredi', 'GNO', 'Okul'])
print(type(pandas_serisi))
print(pandas_serisi)

even_num = pd.Series([i for i in range(0,20,2)])
print(type(even_num))
print(even_num)

list_1 = ['Dogukan', 'Ahmet', 'Ayşe', 'Ali', 'Fatma']
list_2 = [f'Eleman {i}' for i in range(1,6)]

pandas_serisi_1 = pd.Series(data=list_1, index=list_2)
print(pandas_serisi_1)
print('\n***************************\n')
#Seri de değer alma ve değiştirme.
print(pandas_serisi)
print(pandas_serisi[0])#ismi aldık
print(pandas_serisi.loc['İsim'])#ismi böylede alabiliriz
####################################################
pandas_serisi[0] = 'Ahmet'
print(pandas_serisi)#değişiklik yapıldı
pandas_serisi.loc['İsim'] = 'Doğukan'
print(pandas_serisi)
#Eğer belli bir satırı silmek istersek.
yeni_pandas_serisi = pandas_serisi.drop('GNO')#GNO satırını sildik yeni pandas serisine atadık.
print(pandas_serisi, yeni_pandas_serisi, sep='\n')
pandas_serisi.drop('Okul', inplace=True)#Asıl serieden Okul satırını sildi.
print(pandas_serisi)
#####################################################################
print('\n***************************\n')
print('\n***************************\n')
print('\n***************************\n')
#####################################################################
#####################################################################

#Şimdi pandas da dataframe yapısına bakalım.

veri_tablosu = {
    'İsim' : ['Dogukan', 'Ahmet', 'Ayşe', 'Ali', 'Fatma'],
    'Kredi' : [11, 15, 20, 8, 12],
    'GNO' : [3.5, 3.3, 2.8, 2.4, 3.0],
    'Okul' : ['ESTU', 'ODTÜ', 'İTÜ', 'YTÜ', 'Sabancı']
}
print(veri_tablosu)
print(type(veri_tablosu))#şimdi veri_tablosunu daha okunaklı yapalım.

df = pd.DataFrame(veri_tablosu)
print(df)
print(type(df))
#Peki index numaralarının gözükmesini istemezsek ne yaparız ?

yeni_df = pd.DataFrame(veri_tablosu, index=[f'Ogrenci {i}'for i in range(1,6)])
print(yeni_df)#Elimizde daha güzel bir tablo var artık.
print(type(yeni_df))

#Pandas dataframe de değer alma ve değiştirmeye bakalım.
print(yeni_df.loc['Ogrenci 2', 'İsim'])
yeni_df.loc['Ogrenci 2', 'İsim'] = 'Mehmet'
print(yeni_df)#Ahmet ismi Mehmetle değişti
#Birden fazla değer alma ve değiştirme.
print(yeni_df[['İsim', 'GNO']])#Sadece isim ve GNO tabloda gözüktü.
print(yeni_df.loc['Ogrenci 1'])#Ogrenci 1 in tüm bilgileri değişsin.Type Serie ye döndü.
yeni_df.loc['Ogrenci 1'] = ['Batuhan', 28, 2.9, 'Istanbul']
print(yeni_df)#Şimdi tablomuza tekrar bakalım.
print(type(df['İsim']))
#Kısaca colmn lara ulaşırken df[], satırlara ulaşırken df.loc[]
##############################################################
print('\n***************************\n')
print('\n***************************\n')
print('\n***************************\n')
##############################################################
#Daha büyük veriler de naparız peki ?

big_df = pd.read_csv('../weatherHistory.csv')
print(big_df)#Elimizde çok büyük bir tablo var.
print(big_df.columns)#Görüldüğü gibi 12 tane colmn var.Biz sadece Formatted Date ve Temperatureyi alalım.

print(big_df[['Formatted Date', 'Temperature (C)']])#Şimdide son 20 günün sıcaklık verilerini alalım.

big_df_duz = big_df[['Formatted Date', 'Temperature (C)']]
print(big_df_duz.tail(20))
son_20 = big_df_duz.tail(20).copy()
son_20.index = [f'Gun{i}' for i in range(1,21)]#Dataframin indexini değiştirdik.


# 6 ile 12. günler arasını almak istersek
print(son_20.loc['Gun6':'Gun12'])
# Veri değiştirelim
print(son_20)
print(son_20.loc['Gun1','Temperature (C)'])#Gun1 ' in sıcaklıgını değiştirelim.
son_20.loc['Gun1','Temperature (C)'] = 40
print(son_20)
#Peki tüm temperature değerlerini değiltirsek ?
son_20['Temperature (C)'] = [random.randint(20,30) for i in range(1,21)]#Tüm temperature değerleri değişti..
print(son_20)

#Dataframe de drop kullanalım ...
print(big_df.columns)
big_df.drop(['Formatted Date','Summary', 'Precip Type', 'Temperature (C)','Apparent Temperature (C)',
              'Humidity','Wind Speed (km/h)'], inplace=True, axis=1)#Column sildik !
print(big_df)
print(big_df.columns)

