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
fullPathTest = "D:/desktop/TLSB/FirstPyNN/FirstPyNN/DATASET/test/"

"""
HOLA_VALUE = 0
AUTO_VALUE = 1
CAFE_VALUE = 2
ADIOS_VALUE = 3
GRACIAS_VALUE = 4
CBBA_VALUE = 5
CUAL_VALUE = 6
POR_FAVOR_VALUE = 7
QUERER_VALUE = 8
YO_VALUE = 9
"""

HOLA_VALUE = 0
AUTO_VALUE = 1
CAFE_VALUE = 2
ADIOS_VALUE = 3
GRACIAS_VALUE = 0
CBBA_VALUE = 1
CUAL_VALUE = 2
POR_FAVOR_VALUE = 3
QUERER_VALUE = 4
YO_VALUE = 9

HOLA = "HOLA"
AUTO = "AUTO"
CAFE = "CAFE"
ADIOS = "ADIOS"
GRACIAS = "GRACIAS"
CBBA = "CBBA"
CUAL = "CUAL"
POR_FAVOR = "POR_FAVOR"
QUERER = "QUERER"
YO = "YO"

def main(flatten = False):
    #folders = [HOLA, AUTO, CAFE, ADIOS, GRACIAS, CBBA, CUAL, POR_FAVOR, QUERER, YO]
    folders = [GRACIAS, CBBA, CUAL, POR_FAVOR, QUERER]
    train_labels = []
    switcher = {
        HOLA: HOLA_VALUE, 
        AUTO: AUTO_VALUE, 
        CAFE: CAFE_VALUE, 
        ADIOS: ADIOS_VALUE, 
        GRACIAS: GRACIAS_VALUE, 
        CBBA: CBBA_VALUE, 
        CUAL: CUAL_VALUE, 
        POR_FAVOR: POR_FAVOR_VALUE, 
        QUERER: QUERER_VALUE, 
        YO: YO_VALUE
    }

    train, train_labels = getDataset(fullPathTrain, folders, switcher, flatten)
    
    print(train.shape)
    print(train_labels)

    test, test_labels = getDataset(fullPathTest, folders, switcher, flatten)
    
    print(test.shape)
    print(test_labels)
    overAllHistory = []

    test_model = models()
    for index in range(0, len(test_model)):
        model, epochs = test_model[index]
        overAllHistory.append("--------------------------------------------------------------------------------------------------------------------")
        overAllHistory.append("test No: " + str(index))
        results = []
        for y in range(0, 10):
            modelAccuracy, history =  trainModel(model, train, train_labels, test, test_labels, epochs)
            results.append((modelAccuracy, history.history))
        overAllHistory.append(results)

    formatArray(overAllHistory)

def formatArray(historyArray):
    for token in historyArray:
        if(isinstance(token, list)):
            for history in token:
                formatToken(history)
        else:
            print(token)

def formatToken(token):
    accuracy, history = token
    print("Test accuracy: " + str(accuracy))
    print(history)
    print("")

def models():
    models = []

    model0 = keras.Sequential([
        keras.layers.Flatten(input_shape=(1500, 300)),
        keras.layers.Dense(32, activation=tf.nn.sigmoid),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    model1 = keras.Sequential([
        keras.layers.Flatten(input_shape=(1500, 300, 3)),
        keras.layers.Dense(16, activation=tf.nn.sigmoid),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    model2 = keras.Sequential([
        keras.layers.Flatten(input_shape=(1500, 300, 3)),
        keras.layers.Dense(16, activation=tf.nn.sigmoid),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    model3 = keras.Sequential([
        keras.layers.Flatten(input_shape=(1500, 300, 3)),
        keras.layers.Dense(8, activation=tf.nn.sigmoid),
        keras.layers.Dense(8, activation=tf.nn.sigmoid),
        keras.layers.Dense(8, activation=tf.nn.relu),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    models.append((model0, 15))
    models.append((model0, 40))
    models.append((model1, 30))
    models.append((model2, 60))
    models.append((model3, 80))
    models.append((model3, 160))

    return models

def trainModel(model, trainSet, trainLabels, testSet, testLabels, epochs):
    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.summary()

    history = model.fit(trainSet, trainLabels, epochs=epochs)
    test_loss, test_acc = model.evaluate(testSet, testLabels)
#
    print('Test accuracy:', test_acc)

    return test_acc, history

def chargeFolderContent(dataset, path, labels, value, flatten):
    for filename in os.listdir(path):
        img = cv2.imread(path+"/"+filename)
        if img is not None:
            if(flatten):
                dataset.append(toBinarySet(img))
            else:
                dataset.append(img)
            labels.append(value)

def toBinarySet(img):
    h, w = img.shape[:2]
    binarySet = [[0 for x in range(w)] for y in range(h)] 
    for i in range(0, h):
        for j in range(0, w):
            r = int(img[i][j][0])
            g = int(img[i][j][1])
            b = int(img[i][j][2])
            binaryValue = 1 if (r>0 or g>0 or b>0) else 0
            binarySet[i][j] = binaryValue
    return np.array(binarySet)

def grayScale(photo_data):
    photo_data[:] = np.max(photo_data,axis=-1,keepdims=1)/2+np.min(photo_data,axis=-1,keepdims=1)/2

def getDataset(path, folders, switcher, flatten):
    dataset = []
    labels = []

    for folder in folders:
        value = switcher.get(folder, -1)
        chargeFolderContent(dataset, path+folder, labels, value, flatten)
    
    finalDataset = np.array(dataset)
    finalDataset = finalDataset.astype(float) / 255.
    
    return finalDataset, labels

main(flatten=True)