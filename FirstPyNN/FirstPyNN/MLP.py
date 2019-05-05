import keras
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import tensorflow as tf

fullPath = "C:/Users/Jp.SANDRO-HP/Desktop/TLSB/FirstPyNN/FirstPyNN/DATASET/train/"
BANO_VALUE = 1
BUENOS_DIAS_VALUE = 2
HOLA_VALUE = 3
LUZ_VALUE = 4

BANO = "BANO"
BUENOS_DIAS = "BUENOS_DIAS"
HOLA = "HOLA"
LUZ = "LUZ"

def main(flatten):
    folders = [BANO, BUENOS_DIAS, HOLA, LUZ]
    dataset = []
    train_labels = []
    switcher = {
        BANO: BANO_VALUE,
        BUENOS_DIAS: BUENOS_DIAS_VALUE,
        HOLA: HOLA_VALUE,
        LUZ: LUZ_VALUE
    }

    for folder in folders:
        value = switcher.get(folder, -1)
        chargeFolderContent(dataset, folder, train_labels, value, flatten)
    
    train = np.array(dataset)
    train = train.astype(float) / 255.

    if not flatten :
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train[i], cmap=plt.cm.binary)
            #plt.xlabel(class_names[train_labels[i]])
        plt.show()

    print(train.shape)
    print(train_labels)
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(1500, 300, 3)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(train, train_labels, epochs=50)

def chargeFolderContent(dataset, folder, labels, value, flatten):
    path = fullPath + folder;
    for filename in os.listdir(path):
        img = cv2.imread(path+"/"+filename)
        if img is not None:
            if(flatten):
                dataset.append(toBinaryArray(img))
            else:
                dataset.append(img)
            labels.append(value)

def toBinaryArray(npArray):
    binaryArray = [];
    for x in np.nditer(npArray):
        binaryArray.append(x if x>0 else 0)
    return binaryArray

main(False)