import cv2
from os import listdir
from os.path import isfile, join
from edgeDetector import EdgeDetector
import util


class ImageProcessor:
    pictureSize = None
    JPG_EXTENSION = "jpg"

    def __init__(self, pictureSize):
        self.pictureSize = pictureSize
        self.edgeDetector = EdgeDetector(pictureSize)

    def is_given_file_jpg_file(self, file):
        final_token = util.getLastTokenOfPath(file)
        return final_token[1] == self.JPG_EXTENSION

    @staticmethod
    def is_given_path_a_dir(path):
        final_token = util.getLastTokenOfPath(path)
        return len(final_token) == 1

    def gray_scale_images_from(self, path):
        if self.is_given_path_a_dir(path):
            imageFiles = self.get_image_files_from_directory(path)
            for imageFile in imageFiles:
                img = cv2.imread(path + "/" + imageFile, -1)
                img = self.edgeDetector.toGrayscale(img)
                cv2.imwrite(self.combine_name(path + "/" + imageFile, "grey"), img)
        else:
            img = cv2.imread(path, -1)
            img = self.edgeDetector.toGrayscale(img)
            cv2.imwrite(self.combine_name(path, "grey"), img)

    def blurred_edge_images_from(self, path, k_h, k_w):
        if self.is_given_path_a_dir(path):
            image_files = self.get_image_files_from_directory(path)
            for imageFile in image_files:
                img = cv2.imread(path + "/" + imageFile, -1)
                img = self.edgeDetector.getImageBluryEdgesFromNumpy(img, kernelHeight=k_h, kernelWidth=k_w)
                cv2.imwrite(self.combine_name(path + "/" + imageFile, "edges"), img)
        else:
            img = cv2.imread(path, -1)
            img = self.edgeDetector.getImageBluryEdgesFromNumpy(img, kernelHeight=k_h, kernelWidth=k_w)
            cv2.imwrite(self.combine_name(path, "edges"), img)

    def edge_images_from(self, path, k_h, kw):
        if self.is_given_path_a_dir(path):
            image_files = self.get_image_files_from_directory(path)
            for imageFile in image_files:
                img = cv2.imread(path + "/" + imageFile, -1)
                img = self.edgeDetector.getImageEdgesFromNumpy(img, kernelHeight=k_h, kernelWidth=kw)
                cv2.imwrite(self.combine_name(path + "/" + imageFile, "edges"), img)
        else:
            img = cv2.imread(path, -1)
            img = self.edgeDetector.getImageEdgesFromNumpy(img, kernelHeight=k_h, kernelWidth=kw)
            cv2.imwrite(self.combine_name(path, "edges"), img)

    def rescale_images_from(self, path, width, height):
        if self.is_given_path_a_dir(path):
            image_files = self.get_image_files_from_directory(path)
            for imageFile in image_files:
                img = cv2.imread(path + "/" + imageFile, -1)
                resized_img = cv2.resize(img, (width, height))
                cv2.imwrite(self.combine_name(path + "/" + imageFile, "rescaled"), resized_img)
        else:
            img = cv2.imread(path, -1)
            resized_img = cv2.resize(img, (width, height))
            cv2.imwrite(self.combine_name(path, "rescaled"), resized_img)

    @staticmethod
    def combine_name(path, to_combine):
        file_name, extension = util.getLastTokenOfPath(path)[:2]
        new_file_name = file_name + to_combine + "." + extension
        path = util.getPathOfVideoDirectory(path)
        return path + new_file_name

    def get_image_files_from_directory(self, path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        image_files = list(filter(lambda x: self.is_given_file_jpg_file(x), files))
        return image_files
