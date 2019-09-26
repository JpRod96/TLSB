from tensorflow import keras
import numpy as np
import cv2
from MLP import MLPTester


def mlp_single_images():
    full_path_train = "D:/desktop/TLSB/FirstPyNN/FirstPyNN/DATASETCBBA/train/"
    full_path_test = "D:/desktop/TLSB/FirstPyNN/FirstPyNN/DATASETCBBA/test/"

    GESTO1_VALUE = 0
    GESTO2_VALUE = 1
    GESTO3_VALUE = 2

    GESTO1 = "gesto1"
    GESTO2 = "gesto2"
    GESTO3 = "gesto3"

    folders = [GESTO1, GESTO2, GESTO3]
    switcher = {
        GESTO1: GESTO1_VALUE,
        GESTO2: GESTO2_VALUE,
        GESTO3: GESTO3_VALUE
    }

    mlp = MLPTester(MLPTester.SINGLE_IMAGE, folders, switcher, full_path_train, full_path_test)
    mlp.in_test_model_single()


def mlp():
    full_path_train = "D:/desktop/TLSB/FirstPyNN/FirstPyNN/DATASETAugmented/train/"
    full_path_test = "D:/desktop/TLSB/FirstPyNN/FirstPyNN/DATASETAugmented/test/"

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

    mlp = MLPTester(MLPTester.IMAGE_STRIP, folders, switcher, full_path_train, full_path_test)
    mlp.start()
    #mlp.in_test_model()


def test():
    GRACIAS = "GRACIAS"
    CBBA = "CBBA"
    CUAL = "CUAL"
    POR_FAVOR = "POR_FAVOR"
    QUERER = "QUERER"

    values = [GRACIAS, CBBA, CUAL, POR_FAVOR, QUERER]

    new_model = keras.models.load_model('testModel.h5')
    new_model.summary()

    img = cv2.imread("D:/desktop/DATASET/CBBA/20190908_185548Edges.jpg", cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread("D:/desktop/DATASET/CUAL/20190908_185032Edges.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("D:/desktop/DATASET/GRACIAS/20190908_185856Edges.jpg", cv2.IMREAD_GRAYSCALE)
    test_image = [img, img1, img2]

    final_data_set = np.array(test_image)
    final_data_set = final_data_set.astype(float) / 255.

    predictions = new_model.predict(final_data_set)

    for prediction in predictions:
        print(values[np.argmax(prediction)])
        print(prediction)


mlp_single_images()
