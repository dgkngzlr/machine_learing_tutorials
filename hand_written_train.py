import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn

(X_train, y_train), (X_test,y_test) = keras.datasets.mnist.load_data()
X_train = X_train / 255 # Input layerin degeri her zaman 0-1 arasında olmalı
X_test = X_test / 255
print("***********************Test-Train Boyutları**************************")
print("X train : ",len(X_train), "images")
print("y train : ",len(y_train), "images")
print("X test : ",len(X_test), "images")
print("y test : ",len(y_test), "images")
print("*********************************************************************")

print("Resimdeki deger : ",y_train[0])
imgplt = plt.imshow(X_train[0]) # Bir örnek gösterdik.
plt.show()

# Arrayin shapei (60000,28,28), ann'e sokamk için flattenlamalıyız her bir matrixi.

X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test),28*28)
print("*********************************************************************")
print("X_train_flatten_shape :",X_train_flattened.shape)
print("y_train_shape :",y_train.shape)
print("X_test_flatten_shape :",X_test_flattened.shape)
print("y_test_shape :",y_test.shape)
print("*********************************************************************")

# Simdi basit bir ann olusturalım 784(input) x 10(output)

model = keras.Sequential([
    keras.Input(shape=28*28),
    keras.layers.Dense(16,activation="relu"),
    keras.layers.Dense(16,activation="relu"),
    keras.layers.Dense(10,activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)
print(model.summary())

model.fit(X_train_flattened, y_train, epochs=5) # Traine basladı



#####################################Tek bir deger icin tahmin yapıldı#########################
prediction = model.predict(np.array([X_test_flattened[0]])) # girdi olarak np.arrayin icinde np.array olarak vermemiz lazım.
                                                            # X_test_flattened[0].reshape(1,-1) de kullanılabilirdi
print("Prediction matrixi : " , prediction)
plt.imshow(X_test[0])
plt.show()
print("Tahmin edilen sayi :",np.argmax(prediction),
      "Tahmin Skoru : ",prediction[0,np.argmax(prediction)]) # En yüksek tahmin degerine sahip olanın indeksini verir.

################################################################################################

prediction = model.predict(X_test_flattened)
counter = 0
y_pred = list()

for i in prediction:
    if(counter <= 10):
          print("Tahmin edilen :",np.argmax(i),
                "Skor :",i[np.argmax(i)],
                "Gercek deger : ",y_test[counter])
    y_pred.append(np.argmax(i))
    counter += 1
    
print("***********************Confusion Matrix***************************")
print("******************************************************************")
cf=confusion_matrix(y_test, y_pred)
print(cf)
plt.figure(figsize = (10,7))
sn.heatmap(cf,annot=True)
plt.show()

print(X_test_flattened.shape)
model.evaluate(X_test_flattened,y_test) # Test tarafının x ve y lerini ver. Testeki genel skoru gör.