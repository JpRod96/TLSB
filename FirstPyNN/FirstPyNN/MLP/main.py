from tensorflow import keras
import numpy as np
import cv2
import MLP

GRACIAS_VALUE = 0
CBBA_VALUE = 1
CUAL_VALUE = 2
POR_FAVOR_VALUE = 3
QUERER_VALUE = 4

GRACIAS = "GRACIAS"
CBBA = "CBBA"
CUAL = "CUAL"
POR_FAVOR = "POR_FAVOR"
QUERER = "QUERER"


# MLP.main(True)


def test():
    values = [GRACIAS, CBBA, CUAL, POR_FAVOR, QUERER]
    new_model = keras.models.load_model('testModel.h5')
    new_model.summary()

    img = cv2.imread("D:/desktop/DATASET/CBBA/20190908_185548Edges.jpg", cv2.IMREAD_GRAYSCALE)
    test = [img]
    final_data_set = np.array(test)
    final_data_set = final_data_set.astype(float) / 255.
    predictions = new_model.predict(final_data_set)

    print(values[np.argmax(predictions[0])])


test()
