from tensorflow import keras
from dataSetCharger import DataSetCharger
import numpy as np
import tensorflow as tf

fullPathTrain = "D:/desktop/TLSB/FirstPyNN/FirstPyNN/DATASET/train/"
fullPathTest = "D:/desktop/TLSB/FirstPyNN/FirstPyNN/DATASET/test/"
charger = DataSetCharger()

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


def main(flag, flatten=False):
    train, train_labels, test, test_labels, models = setup(flag, flatten)
    test_models(models, train, train_labels, test, test_labels)


def setup(flag, flatten):
    if flag:
        train, train_labels, test, test_labels = load_set(flag, flatten)
        models = image_models()
    else:
        train, train_labels, test, test_labels = load_set(flag, flatten)
        models = lbp_models()
    return train, train_labels, test, test_labels, models


def test_models(models, train, train_labels, test, test_labels):
    over_all_history = []
    for index in range(0, len(models)):
        model, epochs = models[index]
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


def load_set(flag, flatten):
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
    if flag:
        train, train_labels = charger.get_custom_image_data_set(fullPathTrain, folders, switcher, flatten)
        test, test_labels = charger.get_custom_image_data_set(fullPathTest, folders, switcher, flatten)
    else:
        train, train_labels = charger.get_custom_lbp_data_set(fullPathTrain, folders, switcher)
        test, test_labels = charger.get_custom_lbp_data_set(fullPathTest, folders, switcher)

    print(train.shape)
    print(train_labels)
    print(test.shape)
    print(test_labels)

    return train, train_labels, test, test_labels


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


def image_models():
    trainable_models = []

    model0 = keras.Sequential([
        keras.layers.Flatten(input_shape=(2500, 500)),
        keras.layers.Dense(32, activation=tf.nn.sigmoid),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    model0.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    model1 = keras.Sequential([
        keras.layers.Flatten(input_shape=(2500, 500)),
        keras.layers.Dense(16, activation=tf.nn.sigmoid),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    model1.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    model2 = keras.Sequential([
        keras.layers.Flatten(input_shape=(2500, 500)),
        keras.layers.Dense(16, activation=tf.nn.sigmoid),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    model2.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
    # parece el mejor
    model3 = keras.Sequential([
        keras.layers.Flatten(input_shape=(2500, 500)),
        keras.layers.Dense(8, activation=tf.nn.sigmoid),
        keras.layers.Dense(8, activation=tf.nn.sigmoid),
        keras.layers.Dense(8, activation=tf.nn.relu),
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


def lbp_models():
    trainable_models = []

    model0 = keras.Sequential([
        keras.layers.Dense(26, input_shape=(34,)),
        keras.layers.Dense(8, activation=tf.nn.sigmoid),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    model0.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    model1 = keras.Sequential([
        keras.layers.Dense(26, input_shape=(34,)),
        keras.layers.Dense(16, activation=tf.nn.sigmoid),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    model1.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    model2 = keras.Sequential([
        keras.layers.Flatten(input_shape=(2500, 500)),
        keras.layers.Dense(16, activation=tf.nn.sigmoid),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    model2.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
    # parece el mejor
    model3 = keras.Sequential([
        keras.layers.Flatten(input_shape=(2500, 500)),
        keras.layers.Dense(8, activation=tf.nn.sigmoid),
        keras.layers.Dense(8, activation=tf.nn.sigmoid),
        keras.layers.Dense(8, activation=tf.nn.relu),
        keras.layers.Dense(5, activation=tf.nn.softmax)
    ])

    model3.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    trainable_models.append((model0, 80))
    trainable_models.append((model0, 160))
    trainable_models.append((model0, 380))
    trainable_models.append((model1, 160))
    trainable_models.append((model1, 380))
    trainable_models.append((model1, 760))

    return trainable_models


def train_model(model, train_set, train_labels, test_set, test_labels, epochs, weights):
    shuffle_weights(model, weights=weights)
    model.summary()

    history = model.fit(train_set, train_labels, epochs=epochs)
    test_loss, test_acc = model.evaluate(test_set, test_labels)

    print('Test accuracy:', test_acc)

    return test_acc, history


def shuffle_weights(model, weights=None):
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)


main(True)
