from numpy.lib.function_base import diff
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
def farkli_bastir(arr):
    diff_list = list()
    for i in arr:
        if len(diff_list) == 0:
            diff_list.append(i)
            continue
        if i not in diff_list:
           diff_list.append(i)
    
    print("ARRAY FARKLI: ",diff_list)        
 # Araçların kabul edilebilirliğinin 6 attr '  a gore sınıflandırılması.
data = pd.read_csv(".\\data\\car.csv")
min_max_scaler = preprocessing.MinMaxScaler()
SIZE = 1728
ATTR_SIZE = 7

# buying price -- buying  : v-high, high, med, low
buy_price = data.iloc[:, [0]]
buy_price = buy_price.replace({'vhigh': 4, 'high': 3, 'med': 2, 'low': 1})# Buy_price i tutan data_frame. Buna max-min normalizasyonu yapalım.
buy_price_arr = buy_price.iloc[:,0].to_numpy()
buy_price_arr_normalized = min_max_scaler.fit_transform(buy_price_arr.reshape(-1,1))
buy_price_arr_normalized = buy_price_arr_normalized.reshape(1,-1)[0] # 0-1 arasına sıkıstırdık

# price of the maintenance --maint : v-high, high, med, low
maint_price = data.iloc[:, [1]]
maint_price = maint_price.replace({'vhigh': 4, 'high': 3, 'med': 2, 'low': 1})# maint_price i tutan data_frame. Buna max-min normalizasyonu yapalım.
maint_price_arr = maint_price.iloc[:,0].to_numpy()
maint_price_arr_normalized = min_max_scaler.fit_transform(maint_price_arr.reshape(-1,1))
maint_price_arr_normalized = maint_price_arr_normalized.reshape(1,-1)[0]

# doors -- doors  : 2, 3, 4, 5more
doors = data.iloc[:, [2]]
doors = doors.replace({'5more': 6})# doors i tutan data_frame. Buna max-min normalizasyonu yapalım.
doors_arr = doors.iloc[:,0].to_numpy()
doors_arr_normalized = min_max_scaler.fit_transform(doors_arr.reshape(-1,1))
doors_arr_normalized = doors_arr_normalized.reshape(1,-1)[0]

# persons -- persons : 2, 4, more
persons = data.iloc[:, [3]]
persons = persons.replace({'more': 6})
persons_arr = persons.iloc[:,0].to_numpy()
persons_arr_normalized = min_max_scaler.fit_transform(persons_arr.reshape(-1,1))
persons_arr_normalized = persons_arr_normalized.reshape(1,-1)[0]

# the size of luggage boot -- lug_boot : small, med, big
lug_boot = data.iloc[:, [4]]
lug_boot = lug_boot.replace({'small': 0, 'med':1, 'big':2})
lug_boot_arr = lug_boot.iloc[:,0].to_numpy()
lug_boot_normalized = min_max_scaler.fit_transform(lug_boot_arr.reshape(-1,1))
lug_boot_normalized = lug_boot_normalized.reshape(1,-1)[0]

# estimated safety of the car -- safety : low, med, high
safety = data.iloc[:, [5]]
safety = safety.replace({'low': 0, 'med':1, 'high':2})
safety_arr = safety.iloc[:,0].to_numpy()
safety_arr_normalized = min_max_scaler.fit_transform(safety_arr.reshape(-1,1))
safety_arr_normalized = safety_arr_normalized.reshape(1,-1)[0]
farkli_bastir(safety_arr_normalized)
# class car acceptability -- output
accept = data.iloc[:, [6]]
accept = accept.replace({'unacc': 0, 'acc':1, 'good':2,'vgood':3})
accept_arr = accept.iloc[:,0].to_numpy()

# TAHMİN EDİLCEK DEGER DE NORMALIZATION YOK ! (y_train,y_test)


# Tum attrları numpy arrayde birleştirelim.
numpy_data_set = np.zeros((SIZE,ATTR_SIZE))

for i in range(SIZE):
    for j in range(ATTR_SIZE):
        if j == 0 :
            numpy_data_set[i,j]=buy_price_arr_normalized[i]
        
        if j == 1 :
            numpy_data_set[i,j]=maint_price_arr_normalized[i]
        
        if j == 2 :
            numpy_data_set[i,j]=doors_arr_normalized[i]
        
        if j == 3 :
            numpy_data_set[i,j]=persons_arr_normalized[i]
        
        if j == 4 :
            numpy_data_set[i,j]=lug_boot_normalized[i]
        
        if j == 5 :
            numpy_data_set[i,j]=safety_arr_normalized[i]
            
        if j == 6 :
            numpy_data_set[i,j]=accept_arr[i]
print("***********************************NUMPY DATA SET*********************************************")
print(numpy_data_set.shape)
print(numpy_data_set)
print("**********************************************************************************************")
train_set, test_set = train_test_split(numpy_data_set, test_size=0.2, random_state=42)

X_train = train_set[:,:6]
y_train = train_set[:,6]
X_test = test_set[:,:6]
y_test = test_set[:,6]

print("X_train shape :",X_train.shape)
print("y_train shape :",y_train.shape)
print("X_test shape :",X_test.shape)
print("y_test shape :",y_test.shape)

model = keras.Sequential([
    keras.Input(shape=6),
    keras.layers.Dense(16,activation="relu"),
    keras.layers.Dense(32,activation="relu"),
    keras.layers.Dense(32,activation="relu"),
    keras.layers.Dense(16,activation="relu"),
    keras.layers.Dense(4,activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)
print(model.summary())

model.fit(X_train, y_train, epochs=10) # Traine basladı

model.evaluate(X_test,y_test) # Test tarafının x ve y lerini ver. Testeki genel skoru gör.

predictions = model.predict(X_test) # Olasılıklarla birlikte

y_pred = list()
for prd in predictions:
    y_pred.append(np.argmax(prd))
y_pred = np.array(y_pred)

print("***********************Confusion Matrix***************************")
print("******************************************************************")
cf=confusion_matrix(y_test, y_pred)
plt.figure(figsize = (10,7))
graph = sn.heatmap(cf,annot=True)
plt.ylabel("True Value")
plt.xlabel("Pred Value")
plt.show()




#Kullanım
""" 
    buying price  [1.0, 0.6666666666666667, 0.3333333333333333, 0.0]
    maint prcie [1.0, 0.6666666666666667, 0.3333333333333333, 0.0]
    doors [0.0, 0.25, 0.5, 1.0]
    person [0.0, 0.5, 1.0]
    lug_boot [0.0, 0.5, 1.0]
    safety [0.0, 0.5, 1.0]
"""

# Tek bir value için tahmin yapalım 
# O tek valuyu da normalization cercevesinde vermen lazım !, tutarlı olamsı için kodun
tahmin_edilcek_deger = np.array([[1,0.66,1,1,0.4,0.5]],dtype = np.float32)
print(tahmin_edilcek_deger)
result = model.predict(tahmin_edilcek_deger)
print(np.argmax(result))


