import keras
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import tensorflow as tf

def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    print(type(X_train))
    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.
    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])
    return X_train, y_train, X_val, y_val, X_test, y_test

def loadMyDataset(flatten):
    path = "C:/Users/Jp.SANDRO-HP/Desktop/TLSB/FirstPyNN/FirstPyNN/DATASET/train/BAÑO/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    dataset = [];
    for file in files:
        filePath = path + file
        image = cv2.imread(file, 1)
        if(flatten):
            dataset.append(toBinaryArray(image))
        else:
            dataset.append(image)

    train = np.array(dataset)
    train = train.astype(float) / 255.
    train_labels = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

    #if not flatten :
    #    plt.figure(figsize=(10,10))
    #    for i in range(5):
    #        plt.subplot(5,5,i+1)
    #        plt.xticks([])
    #        plt.yticks([])
    #        plt.grid(False)
    #        plt.imshow(train[i], cmap=plt.cm.binary)
    #        #plt.xlabel(class_names[train_labels[i]])
    #    plt.show()

    print(train.shape)
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(1500, 300, 3)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(train, train_labels, epochs=5)

def toBinaryArray(npArray):
    binaryArray = [];
    for x in np.nditer(npArray):
        binaryArray.append(x if x>0 else 0)
    return binaryArray

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    ## Printing dimensions
    print(X_train.shape, y_train.shape)
    ## Visualizing the first digit
    plt.figure()
    plt.imshow(X_train[0], cmap="Greys")
    plt.colorbar()
    plt.grid(False)
    plt.show()
    print(y_train[0])

    ## Changing dimension of input images from N*28*28 to  N*784
    X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
    print('Train dimension:');print(X_train.shape)
    print('Test dimension:');print(X_test.shape)
    ## Changing labels to one-hot encoded vector
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)
    print('Train labels dimension:');print(y_train.shape)
    print('Test labels dimension:');print(y_test.shape)
loadMyDataset(False)