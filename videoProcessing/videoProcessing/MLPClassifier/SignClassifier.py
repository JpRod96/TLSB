from tensorflow import keras
import numpy as np
import cv2


class MLPSignClassifier:
    GESTO_1 = "GESTO1"
    GESTO_2 = "GESTO2"
    GESTO_3 = "GESTO3"

    def __init__(self, model_path):
        self.classes = [self.GESTO_1, self.GESTO_2, self.GESTO_3]

        self.model = keras.models.load_model(model_path)
        print("Model charged successfully")
        self.model.summary()

    def predict(self, image):
        to_predict = [image]
        final_data_set = np.array(to_predict)
        final_data_set = final_data_set.astype(float) / 255.

        prediction = self.model.predict(final_data_set)
        print(self.classes[np.argmax(prediction)])
        print(prediction)
