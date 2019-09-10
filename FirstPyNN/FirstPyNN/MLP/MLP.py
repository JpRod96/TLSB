from tensorflow import keras
from dataSetCharger import DataSetCharger
import numpy as np
import tensorflow as tf

charger = DataSetCharger()
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


def main(flatten=False):
    # folders = [HOLA, AUTO, CAFE, ADIOS, GRACIAS, CBBA, CUAL, POR_FAVOR, QUERER, YO]
    folders = [GRACIAS, CBBA, CUAL, POR_FAVOR, QUERER]
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

    train, train_labels = charger.get_data_set(fullPathTrain, folders, switcher, flatten)

    print(train.shape)
    print(train_labels)

    test, test_labels = charger.get_data_set(fullPathTest, folders, switcher, flatten)

    print(test.shape)
    print(test_labels)
    over_all_history = []

    test_model = models()
    for index in range(0, len(test_model)):
        model, epochs = test_model[index]
        weights = model.get_weights()
        over_all_history.append(
            "--------------------------------------------------------------------------------------------------------")
        over_all_history.append("test No: " + str(index))
        results = []
        for y in range(0, 15):
            model_accuracy, history = train_model(model, train, train_labels, test, test_labels, epochs, weights)
            results.append((model_accuracy, history.history))
        over_all_history.append(results)

    save_test_to_txt(over_all_history)


def save_test_to_txt(history_array):
    f = open("testResults.txt", "w+")
    f.write(format_array(history_array))
    f.close()


def format_array(history_array):
    string = ""
    for token in history_array:
        if isinstance(token, list):
            for history in token:
                string += format_token(history)
        else:
            string += token + "\n"
    return string


def format_token(token):
    string = ""
    accuracy, history = token
    string += "Test accuracy: " + str(accuracy) + "\n"
    string += str(history) + "\n\n"
    return string


def models():
    trainable_models = []

    model0 = keras.Sequential([
        keras.layers.Flatten(input_shape=(2500, 500)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, activation=tf.nn.sigmoid),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    model0.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    model1 = keras.Sequential([
        keras.layers.Flatten(input_shape=(2500, 500)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation=tf.nn.sigmoid),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    model1.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    model2 = keras.Sequential([
        keras.layers.Flatten(input_shape=(2500, 500)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation=tf.nn.sigmoid),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    model2.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
    # parece el mejor
    model3 = keras.Sequential([
        keras.layers.Flatten(input_shape=(2500, 500)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(8, activation=tf.nn.sigmoid),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(8, activation=tf.nn.sigmoid),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(8, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    model3.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    trainable_models.append((model0, 15))
    trainable_models.append((model0, 40))
    trainable_models.append((model1, 30))
    trainable_models.append((model1, 60))
    trainable_models.append((model2, 60))
    trainable_models.append((model2, 80))
    trainable_models.append((model3, 80))
    trainable_models.append((model3, 160))

    return trainable_models


def train_model(model, train_set, train_labels, test_set, test_labels, epochs, weights):
    shuffle_weights(model, weights=weights)
    model.summary()

    history = model.fit(train_set, train_labels, epochs=epochs)
    test_loss, test_acc = model.evaluate(test_set, test_labels)

    print('Test accuracy:', test_acc)

    return test_acc, history


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.

    This is a fast approximation of re-initializing the weights of a model.

    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).

    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)

main()
