import os
import cv2
import numpy as np
from skimage import feature


class DataSetCharger:

    def charge_folder_content_as_image_strip(self, data_set, path, labels, value, flatten):
        for filename in os.listdir(path):
            img = cv2.imread(path + "/" + filename, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                if flatten:
                    data_set.append(self.to_binary_set(img))
                else:
                    img = cv2.resize(img, (20, 20))
                    data_set.append(img)
                    print(len(data_set))
                labels.append(value)

    def charge_folder_content_as_lbp(self, data_set, path, labels, value, num_points, radius):
        for filename in os.listdir(path):
            img = cv2.imread(path + "/" + filename, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                print("transforming image")
                data_set.append(self.image_to_lbp(img, num_points, radius))
                print("Done")
                labels.append(value)

    @staticmethod
    def to_binary_set(img):
        print('Transforming input to binary set...')
        h, w = img.shape[:2]
        binary_set = [[0 for x in range(w)] for y in range(h)]
        for i in range(0, h):
            for j in range(0, w):
                if len(img[i][j].shape) > 1:
                    r = int(img[i][j][0])
                    g = int(img[i][j][1])
                    b = int(img[i][j][2])
                    binary_value = 1 if (r > 0 or g > 0 or b > 0) else 0
                else:
                    binary_value = 1 if img[i][j] > 0 else 0
                binary_set[i][j] = binary_value
        print('done')
        return np.array(binary_set)

    def get_custom_image_data_set(self, path, folders, switcher, flatten):
        data_set = []
        labels = []

        for folder in folders:
            value = switcher.get(folder, -1)
            self.charge_folder_content_as_image_strip(data_set, path + folder, labels, value, flatten)

        final_data_set = np.array(data_set)
        final_data_set = final_data_set.astype(float) / 255.

        return final_data_set, labels

    def get_custom_lbp_data_set(self, path, folders, switcher):
        data_set = []
        labels = []
        radius = 3
        n_points = 8 * radius

        for folder in folders:
            value = switcher.get(folder, -1)
            self.charge_folder_content_as_lbp(data_set, path + folder, labels, value, n_points, radius)

        final_data_set = np.array(data_set)
        final_data_set = final_data_set.astype(float) / 255.

        return final_data_set, labels

    @staticmethod
    def get_lbp_set_from_file(path):
        data_set = np.genfromtxt(path + "/values.txt")
        labels = np.genfromtxt(path + "/labels.txt")

        final_data_set = data_set.astype(float) / 255.

        return final_data_set, labels

    def image_to_lbp(self, image, num_points, radius, eps=1e-7):
        return self.describe(image, num_points, radius, eps)

    @staticmethod
    def describe(image, num_points, radius, eps):
        lbp = feature.local_binary_pattern(image, num_points,
                                           radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, num_points + 3),
                                 range=(0, num_points + 2))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist
