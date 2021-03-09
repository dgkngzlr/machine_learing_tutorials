# Udemy 2.7
import pandas as pd

veri = pd.read_csv('veriler.csv')
print(veri)
print(veri[['ulke']])
print(veri['ulke'][0])#Tek veriyi cekmek için.
