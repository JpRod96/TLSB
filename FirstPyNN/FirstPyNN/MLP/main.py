from tensorflow import keras
import numpy as np
import cv2
from MLP import MLPTester


def mlp_single_images():
    full_path_train = "D:/desktop/DIGIT/numbers/UNCATEGORIZED/"
    full_path_test = "D:/desktop/DIGIT/numbers/UNCATEGORIZED/test/"

    GESTO0_VALUE = 0
    GESTO1_VALUE = 1
    GESTO2_VALUE = 2
    GESTO3_VALUE = 3
    GESTO4_VALUE = 4
    GESTO5_VALUE = 5
    GESTO6_VALUE = 6
    GESTO7_VALUE = 7
    GESTO8_VALUE = 8
    GESTO9_VALUE = 9

    GESTO0 = "0"
    GESTO1 = "1"
    GESTO2 = "2"
    GESTO3 = "3"
    GESTO4 = "4"
    GESTO5 = "5"
    GESTO6 = "6"
    GESTO7 = "7"
    GESTO8 = "8"
    GESTO9 = "9"

    folders = [GESTO0, GESTO1, GESTO2, GESTO3, GESTO4, GESTO5, GESTO6, GESTO7, GESTO8, GESTO9]
    switcher = {
        GESTO0: GESTO0_VALUE,
        GESTO1: GESTO1_VALUE,
        GESTO2: GESTO2_VALUE,
        GESTO3: GESTO3_VALUE,
        GESTO4: GESTO4_VALUE,
        GESTO5: GESTO5_VALUE,
        GESTO6: GESTO6_VALUE,
        GESTO7: GESTO7_VALUE,
        GESTO8: GESTO8_VALUE,
        GESTO9: GESTO9_VALUE
    }

    mlp = MLPTester(MLPTester.SINGLE_IMAGE, folders, switcher, full_path_train, full_path_test)
    mlp.in_test_model_single2()


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
    GESTO1 = "gesto1"
    GESTO2 = "gesto2"
    GESTO3 = "gesto3"

    values = [GESTO1, GESTO2, GESTO3]

    new_model = keras.models.load_model('D:/desktop/TLSB/FirstPyNN/FirstPyNN/MLP/bestModel75.h5')
    new_model.summary()

    img = cv2.imread("D:/desktop/TLSB/FirstPyNN/FirstPyNN/DATASETCBBAORIGINAL/train/gesto1/0.jpg", cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread("D:/desktop/TLSB/FirstPyNN/FirstPyNN/DATASETCBBAORIGINAL/train/gesto2/0.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("D:/desktop/TLSB/FirstPyNN/FirstPyNN/DATASETCBBAORIGINAL/train/gesto3/0.jpg", cv2.IMREAD_GRAYSCALE)
    test_image = [img, img1, img2]

    final_data_set = np.array(test_image)
    final_data_set = final_data_set.astype(float) / 255.

    predictions = new_model.predict(final_data_set)

    for prediction in predictions:
        print(values[np.argmax(prediction)])
        print(prediction)


mlp_single_images()
