from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import tensorflow as tf

fullPathTrain = "D:/desktop/TLSB/FirstPyNN/FirstPyNN/DATASET/train/"
fullPathTest = "D:/desktop/TLSB/FirstPyNN/FirstPyNN/DATASET/train/"
BANO_VALUE = 0
BUENOS_DIAS_VALUE = 1
HOLA_VALUE = 2
LUZ_VALUE = 3

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
        chargeFolderContent(dataset, fullPathTrain+folder, train_labels, value, flatten)
    
    train = np.array(dataset)
    train = train.astype(float) / 255.
    
    print(train.shape)
    print(train_labels)

    dataset_test = []
    test_labels = []

    for folder in folders:
        value = switcher.get(folder, -1)
        chargeFolderContent(dataset_test, fullPathTest+folder, test_labels, value, flatten)

    test = np.array(dataset_test)
    test = test.astype(float) / 255.
    
    print(test.shape)
    print(test_labels)

    #if not flatten :
    #    plt.figure(figsize=(10,10))
    #    for i in range(25):
    #        plt.subplot(5,5,i+1)
    #        plt.xticks([])
    #        plt.yticks([])
    #        plt.grid(False)
    #        plt.imshow(train[i], cmap=plt.cm.binary)
    #        #plt.xlabel(class_names[train_labels[i]])
    #    plt.show()

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(1500, 300, 3)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(4, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(train, train_labels, epochs=50)
    test_loss, test_acc = model.evaluate(test, test_labels)
#
    print('Test accuracy:', test_acc)

def chargeFolderContent(dataset, path, labels, value, flatten):
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