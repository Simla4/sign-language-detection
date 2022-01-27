######## LIBRARIES #########
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator


from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


################ PARAMETERS ########################
batchSizeVal = 128
epochsVal = 100
stepsPerEpochVal = 10
####################################################

#### VERİYİ IMPORT EDİYORUZ
df_train = pd.read_csv('train.csv')
print(df_train.head(-5))
df_test = pd.read_csv('test.csv')
print(df_test.head(-5))

#### Örnek resim
plt.figure(figsize = (15,10))
sns.set_style("darkgrid")
sns.countplot(df_train['label'])
plt.show()

#### Verilerin labelleri ayrıldı
y_train = df_train['label']
df_train.drop(['label'], axis=1, inplace=True)
print(df_train.head(-5))
y_test = df_test['label']
df_test.drop(['label'], axis=1, inplace=True)
print(df_test.head(-5))

#### Resimler yeniden boyutlandırrıldı
X_train = df_train.values.reshape(df_train.shape[0], 28, 28, 1)
X_test = df_test.values.reshape(df_test.shape[0], 28, 28, 1)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

#### Data Augmentation
datagen = ImageDataGenerator(rescale=1./255,
                             zoom_range=0.2,
                             width_shift_range=.2, height_shift_range=.2,
                             rotation_range=30,
                             brightness_range=[0.8, 1.2],
                             horizontal_flip=True)

datagenRescale = ImageDataGenerator(rescale=1./255)

X_train = datagen.flow(X_train, y_train, batch_size=batchSizeVal)
X_test = datagenRescale.flow(X_test, y_test)

#### Verilerin labelleri alfabenin harfleri ile eşleştirildi
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#### Eşleşirilen harfler test edildi
plt.figure(figsize=(15, 15))
for i in range(9):
    plt.subplot(3, 3, i+1)
    for X_batch, Y_batch in X_train:
        image = X_batch[i]
        plt.imshow(image, cmap='gray')
        plt.xlabel(alphabet[Y_batch[i]])
        break
plt.show()

####  Setup Callbacks
checkpoint_filepath = 'best_model.hdf5'
callback_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_accuracy', mode='max', save_best_only=True)
callback_learningrate = ReduceLROnPlateau(monitor='loss', mode='min', min_delta=0.01, patience=3, factor=.75, min_lr=0.00001, verbose=1)
callbacks = [callback_checkpoint, callback_learningrate]

#### Neural Network
model = Sequential([Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
                    MaxPooling2D(2, 2, padding='same'),
                    Dropout(0.2),

                    Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
                    MaxPooling2D(2, 2, padding='same'),
                    Dropout(0.2),

                    Conv2D(filters=512, kernel_size=(3, 3), activation="relu"),
                    MaxPooling2D(2, 2, padding='same'),
                    Dropout(0.2),

                    Flatten(),

                    Dense(units=4096, activation="relu"),
                    Dropout(0.2),

                    Dense(units=1024, activation="relu"),
                    Dropout(0.2),

                    Dense(units=256, activation="relu"),
                    Dropout(0.2),

                    Dense(units=25, activation="softmax"),
                    ])

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()

#### Model öğrenmeye başlıyor
history = model.fit(X_train, validation_data=X_test, epochs=epochsVal, callbacks=callbacks)

#### Verinin kayıp değerleri
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

#### Training verilerini görselleştirdik
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#### EĞİTİLEN MODELİ KAYDETTİK

json_file = model.to_json()
with open("model_trained.json", "w") as file:
   file.write(json_file);

model.save_weights("model.h5")