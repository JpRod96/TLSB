import cv2
import os
from os import listdir
from os.path import isfile, join
from edgeDetector import EdgeDetector
import numpy as np
import util


class ImageProcessor:
    pictureSize = None
    JPG_EXTENSION = "jpg"

    def __init__(self, picture_size, path):
        self.pictureSize = picture_size
        self.edgeDetector = EdgeDetector(picture_size)
        self.path = path

    def is_given_file_jpg_file(self, file):
        final_token = util.getLastTokenOfPath(file)
        return final_token[1] == self.JPG_EXTENSION

    def is_given_path_a_dir(self):
        final_token = util.getLastTokenOfPath(self.path)
        return len(final_token) == 1

    def gray_scale_images_from(self):
        if self.is_given_path_a_dir():
            image_files = self.get_image_files_from_directory()
            for imageFile in image_files:
                img = cv2.imread(self.path + "/" + imageFile, -1)
                img = self.edgeDetector.toGrayscale(img)
                cv2.imwrite(self.combine_name(self.path + "/" + imageFile, "grey"), img)
        else:
            img = cv2.imread(self.path, -1)
            img = self.edgeDetector.toGrayscale(img)
            cv2.imwrite(self.combine_name(self.path, "grey"), img)

    def augment_images_from(self, aug, batch_size):
        if self.is_given_path_a_dir():
            image_files = self.get_image_files_from_directory()
            for imageFile in image_files:
                img = cv2.imread(self.path + "/" + imageFile, -1)
                self.transform(img, aug, batch_size, self.path + "/" + imageFile)
        else:
            img = cv2.imread(self.path, -1)
            img = self.edgeDetector.toGrayscale(img)
            cv2.imwrite(self.combine_name(self.path, "grey"), img)

    def transform(self, image, aug, size, name):
        cont = size
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        else:
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=3)
        augmented_image_iterator = aug.flow(image, batch_size=1)
        for augmented_image in augmented_image_iterator:
            if cont > 0:
                image = np.squeeze(augmented_image, axis=0)
                cv2.imwrite(self.combine_name(name, "augmented" + str(cont)), image)
            else:
                break
            cont -= 1

    def blurred_edge_images_from(self, k_h, k_w):
        if self.is_given_path_a_dir():
            image_files = self.get_image_files_from_directory()
            for imageFile in image_files:
                img = cv2.imread(self.path + "/" + imageFile, -1)
                img = self.edgeDetector.getImageBluryEdgesFromNumpy(img, kernelHeight=k_h, kernelWidth=k_w)
                cv2.imwrite(self.combine_name(self.path + "/" + imageFile, "edges"), img)
        else:
            img = cv2.imread(self.path, -1)
            img = self.edgeDetector.getImageBluryEdgesFromNumpy(img, kernelHeight=k_h, kernelWidth=k_w)
            cv2.imwrite(self.combine_name(self.path, "edges"), img)

    def edge_images_from(self, k_h, kw):
        if self.is_given_path_a_dir():
            image_files = self.get_image_files_from_directory()
            for imageFile in image_files:
                img = cv2.imread(self.path + "/" + imageFile, -1)
                img = self.edgeDetector.getImageEdgesFromNumpy(img, kernelHeight=k_h, kernelWidth=kw)
                cv2.imwrite(self.combine_name(self.path + "/" + imageFile, "edges"), img)
        else:
            img = cv2.imread(self.path, -1)
            img = self.edgeDetector.getImageEdgesFromNumpy(img, kernelHeight=k_h, kernelWidth=kw)
            cv2.imwrite(self.combine_name(self.path, "edges"), img)

    def rescale_images_from(self, width=0, height=0):
        if width is 0:
            width = self.pictureSize
        if height is 0:
            height = self.pictureSize
        images = self.load_images_from_path()
        os.mkdir(self.path + "/rescaled")
        cont = 0
        for image in images:
            resized_img = cv2.resize(image, (width, height))
            cv2.imwrite(self.path + "/rescaled/" + str(cont) + ".jpg", resized_img)
            cont += 1

    def load_images_from_path(self):
        files = []
        if self.is_given_path_a_dir():
            image_files = self.get_image_files_from_directory()
            for imageFile in image_files:
                img = cv2.imread(self.path + "/" + imageFile, -1)
                if img is not None:
                    files.append(img)
        else:
            img = cv2.imread(self.path, -1)
            if img is not None:
                files.append(img)
        return files

    @staticmethod
    def combine_name(path, to_combine):
        file_name, extension = util.getLastTokenOfPath(path)[:2]
        new_file_name = file_name + to_combine + "." + extension
        path = util.getPathOfVideoDirectory(path)
        return path + new_file_name

    def get_image_files_from_directory(self):
        files = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        image_files = list(filter(lambda x: self.is_given_file_jpg_file(x), files))
        return image_files
