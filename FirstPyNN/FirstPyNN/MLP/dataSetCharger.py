import os
import cv2
import numpy as np
from skimage import feature


class DataSetCharger:
    def charge_folder_content(self, data_set, path, labels, value, flatten):
        for filename in os.listdir(path):
            img = cv2.imread(path + "/" + filename, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if flatten:
                    data_set.append(self.to_binary_set(img))
                else:
                    data_set.append(img)
                labels.append(value)

    @staticmethod
    def to_binary_set(img):
        print('Transforming input to binary set...')
        h, w = img.shape[:2]
        binary_set = [[0 for x in range(w)] for y in range(h)]
        for i in range(0, h):
            for j in range(0, w):
                r = int(img[i][j][0])
                g = int(img[i][j][1])
                b = int(img[i][j][2])
                binary_value = 1 if (r > 0 or g > 0 or b > 0) else 0
                binary_set[i][j] = binary_value
        print('done')
        return np.array(binary_set)

    def get_data_set(self, path, folders, switcher, flatten):
        data_set = []
        labels = []

        for folder in folders:
            value = switcher.get(folder, -1)
            self.charge_folder_content(data_set, path + folder, labels, value, flatten)

        final_data_set = np.array(data_set)
        final_data_set = final_data_set.astype(float) / 255.

        return final_data_set, labels

    def image_to_lbp(self, image_path, num_points, radius, eps=1e-7):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return self.describe(image, num_points, radius, eps)

    def describe(self, image, num_points, radius, eps):
        lbp = feature.local_binary_pattern(image, num_points,
                                           radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, num_points + 3),
                                 range=(0, num_points + 2))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist
