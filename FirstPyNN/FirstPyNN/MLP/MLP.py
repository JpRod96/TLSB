from tensorflow import keras
from dataSetCharger import DataSetCharger
import numpy as np
import tensorflow as tf


class MLPTester:
    IMAGE_STRIP = "strip"
    SINGLE_IMAGE = "single"
    LBP = "lbp"
    IN_TEST = "in_test"

    def __init__(self, use_image_set, folders, switcher, path_train, path_test, flatten=False):
        self.flag = use_image_set
        self.folders = folders
        self.switcher = switcher
        self.path_train = path_train
        self.path_test = path_test
        self.flatten = flatten
        self.charger = DataSetCharger()
        self.best_score = 0

    def start(self):
        train, train_labels, test, test_labels, models = self.setup()
        self.test_models(models, train, train_labels, test, test_labels)

    def setup(self):
        if self.flag is self.IMAGE_STRIP:
            models = self.image_models()
        elif self.flag is self.LBP:
            models = self.lbp_models()
        else:
            models = self.single_image_models()

        train, train_labels, test, test_labels = self.load_set()
        return train, train_labels, test, test_labels, models

    def test_models(self, models, train, train_labels, test, test_labels):
        over_all_history = []
        for index in range(0, len(models)):
            model, epochs = models[index]
            weights = model.get_weights()
            over_all_history.append(
                "--------------------------------------------------------------------------------------------------------")
            over_all_history.append("test No: " + str(index))
            string_list = []
            model.summary(print_fn=lambda x: string_list.append(x))
            short_model_summary = "\n".join(string_list)
            over_all_history.append(short_model_summary)
            results = []
            for y in range(0, 10):
                model_accuracy, history = self.train_model(model, train, train_labels, test, test_labels, epochs,
                                                           weights)
                results.append((model_accuracy, history.history))
            over_all_history.append(results)

        self.save_test_to_txt(over_all_history)

    def load_set(self):
        if self.flag == self.IMAGE_STRIP or self.flag == self.SINGLE_IMAGE:
            train, train_labels = self.charger.get_custom_image_data_set(self.path_train, self.folders, self.switcher,
                                                                         self.flatten)
            test, test_labels = self.charger.get_custom_image_data_set(self.path_test, self.folders, self.switcher,
                                                                       self.flatten)
        else:
            train, train_labels = self.charger.get_custom_lbp_data_set(self.path_train, self.folders, self.switcher)
            test, test_labels = self.charger.get_custom_lbp_data_set(self.path_test, self.folders, self.switcher)

        print(train.shape)
        print(train_labels)
        print(test.shape)
        print(test_labels)

        return train, train_labels, test, test_labels

    def save_test_to_txt(self, history_array):
        f = open("testResults.txt", "w+")
        f.write(self.format_array(history_array))
        f.close()

    def format_array(self, history_array):
        string = ""
        for token in history_array:
            if isinstance(token, list):
                for history in token:
                    string += self.format_token(history)
            else:
                string += token + "\n"
        return string

    @staticmethod
    def format_token(token):
        string = ""
        accuracy, history = token
        string += "Test accuracy: " + str(accuracy) + "\n"
        string += str(history) + "\n\n"
        return string

    def in_test_model(self):
        train, train_labels, test, test_labels = self.load_set()
        over_all_history = []
        activation_functions = [tf.nn.sigmoid, tf.nn.relu]
        cont = 1
        for neurons_outer in range(16, 20):
            for neurons_inner in range(16, 20):
                for activation_function in activation_functions:
                    for iteration in [15, 20, 50, 80]:
                        model = keras.Sequential([
                            keras.layers.Flatten(input_shape=(1500, 300)),
                            keras.layers.Dense(neurons_outer, activation=activation_function),
                            keras.layers.Dense(neurons_inner, activation=activation_function),
                            keras.layers.Dense(5, activation=tf.nn.softmax)
                        ])
                        model.compile(optimizer='adam',
                                      loss='sparse_categorical_crossentropy',
                                      metrics=['accuracy'])
                        weights = model.get_weights()
                        over_all_history.append(
                            "--------------------------------------------------------------------------------------------------------")
                        over_all_history.append("test No: " + str(cont))
                        string_list = []
                        model.summary(print_fn=lambda x: string_list.append(x))
                        short_model_summary = "\n".join(string_list)
                        over_all_history.append(short_model_summary)
                        if activation_function is tf.nn.relu:
                            over_all_history.append("Rectilineo \n")
                            print("rectilineo")
                        else:
                            over_all_history.append("Sigmoidal \n")
                            print("sigmoidal")
                        results = []
                        model_accuracy, history = self.train_model(model, train, train_labels, test, test_labels,
                                                                   iteration,
                                                                   weights)
                        results.append((model_accuracy, history.history))
                        over_all_history.append(results)
                        cont += 1
        self.save_test_to_txt(over_all_history)

    def in_test_model_single(self):
        train, train_labels, test, test_labels = self.load_set()
        over_all_history = []
        activation_functions = [tf.nn.sigmoid, tf.nn.relu, tf.nn.tanh]
        cont = 1
        for neurons_outer in range(5, 25):
            for activation_function in activation_functions:
                for iteration in [10, 15, 20, 30, 50, 80, 100, 120, 140]:
                    model = keras.Sequential([
                        keras.layers.Flatten(input_shape=(300, 300)),
                        keras.layers.Dense(neurons_outer, activation=activation_function),
                        keras.layers.Dense(3, activation=tf.nn.softmax)
                    ])
                    model.compile(optimizer='adam',
                                  loss='sparse_categorical_crossentropy',
                                  metrics=['accuracy'])
                    weights = model.get_weights()
                    over_all_history.append(
                        "--------------------------------------------------------------------------------------------------------")
                    over_all_history.append("test No: " + str(cont))
                    over_all_history.append("iteraciones: " + str(iteration))
                    string_list = []
                    model.summary(print_fn=lambda x: string_list.append(x))
                    short_model_summary = "\n".join(string_list)
                    over_all_history.append(short_model_summary)
                    if activation_function is tf.nn.relu:
                        over_all_history.append("Rectilineo \n")
                        print("rectilineo")
                    else:
                        over_all_history.append("Sigmoidal \n")
                        print("sigmoidal")
                    results = []
                    for index in range(0, 10):
                        model_accuracy, history = self.train_model(model, train, train_labels, test, test_labels,
                                                                   iteration,
                                                                   weights)
                        results.append((model_accuracy, history.history))
                    over_all_history.append(results)
                    cont += 1
        self.save_test_to_txt(over_all_history)

    @staticmethod
    def image_models():
        trainable_models = []
        model0 = keras.Sequential([
            keras.layers.Flatten(input_shape=(1500, 300)),
            keras.layers.Dense(17, activation=tf.nn.sigmoid),
            keras.layers.Dense(17, activation=tf.nn.sigmoid),
            keras.layers.Dense(5, activation=tf.nn.softmax)
        ])

        model0.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

        model1 = keras.Sequential([
            keras.layers.Flatten(input_shape=(1500, 300)),
            keras.layers.Dense(17, activation=tf.nn.sigmoid),
            keras.layers.Dense(17, activation=tf.nn.sigmoid),
            keras.layers.Dense(5, activation=tf.nn.softmax)
        ])

        model1.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

        model2 = keras.Sequential([
            keras.layers.Flatten(input_shape=(1500, 300)),
            keras.layers.Dense(17, activation=tf.nn.sigmoid),
            keras.layers.Dense(17, activation=tf.nn.sigmoid),
            keras.layers.Dense(5, activation=tf.nn.softmax)
        ])

        model2.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

        model3 = keras.Sequential([
            keras.layers.Flatten(input_shape=(1500, 300)),
            keras.layers.Dense(17, activation=tf.nn.sigmoid),
            keras.layers.Dense(17, activation=tf.nn.sigmoid),
            keras.layers.Dense(5, activation=tf.nn.softmax)
        ])

        model3.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

        trainable_models.append((model0, 80))
        trainable_models.append((model1, 80))
        trainable_models.append((model2, 80))
        trainable_models.append((model3, 80))

        return trainable_models

    @staticmethod
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
            keras.layers.Flatten(input_shape=(1500, 300)),
            keras.layers.Dense(16, activation=tf.nn.sigmoid),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dense(5, activation=tf.nn.softmax)
        ])

        model2.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
        # parece el mejor
        model3 = keras.Sequential([
            keras.layers.Flatten(input_shape=(1500, 300)),
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

    @staticmethod
    def single_image_models():
        trainable_models = []

        model0 = keras.Sequential([
            keras.layers.Flatten(input_shape=(300, 300)),
            keras.layers.Dense(8, activation=tf.nn.sigmoid),
            keras.layers.Dense(3, activation=tf.nn.softmax)
        ])

        model0.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

        model1 = keras.Sequential([
            keras.layers.Flatten(input_shape=(300, 300)),
            keras.layers.Dense(16, activation=tf.nn.sigmoid),
            keras.layers.Dense(3, activation=tf.nn.softmax)
        ])

        model1.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

        model2 = keras.Sequential([
            keras.layers.Flatten(input_shape=(300, 300)),
            keras.layers.Dense(32, activation=tf.nn.sigmoid),
            keras.layers.Dense(3, activation=tf.nn.softmax)
        ])

        model2.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

        trainable_models.append((model0, 20))
        trainable_models.append((model0, 40))
        trainable_models.append((model0, 80))
        trainable_models.append((model1, 20))
        trainable_models.append((model1, 40))
        trainable_models.append((model1, 80))
        trainable_models.append((model2, 40))
        trainable_models.append((model2, 80))
        trainable_models.append((model2, 100))

        return trainable_models

    def train_model(self, model, train_set, train_labels, test_set, test_labels, epochs, weights):
        self.shuffle_weights(model, weights=weights)
        model.summary()

        history = model.fit(train_set, train_labels, epochs=epochs)
        test_loss, test_acc = model.evaluate(test_set, test_labels)

        print('Test accuracy:', test_acc)
        self.save_if_the_best(test_acc, model)

        return test_acc, history

    def save_if_the_best(self, accuracy, model):
        accuracy = int(accuracy * 100)
        if accuracy > self.best_score:
            model.save("bestModel" + str(accuracy) + ".h5")
            self.best_score = accuracy

    @staticmethod
    def shuffle_weights(model, weights=None):
        if weights is None:
            weights = model.get_weights()
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        # Faster, but less random: only permutes along the first dimension
        # weights = [np.random.permutation(w) for w in weights]
        model.set_weights(weights)
