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
    NONE = 0
    GRAY = 1
    EDGES = 2
    BLURRY_EDGES = 3

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

    def get_strip_from(self, strip_length=0, aug=(None, 0), image_filter=0):
        images = self.load_images_from_path()
        aug_processor, augmented_size = aug

        os.mkdir(self.path + "/strips")
        kh, kw = self.on_the_fly_calibration(images[0], image_filter)
        strip = self.satisfy_frames_number(self.apply_filter(images, image_filter, kh=kh, kw=kw), strip_length)
        if aug_processor is None:
            file_name = self.path + "/strips/originalStrip." + self.JPG_EXTENSION
            self.save_strip(strip, file_name)
        else:
            base_file_name = self.path + "/strips/"
            self.augment_image_strip(strip, aug_processor, augmented_size, base_file_name)

    def augment_image_strip(self, images_strip, augment_processor, augmented_size, base_file_name):
        if len(images_strip) > 0:
            print("Augmenting image strip...")

            for x in range(0, augmented_size):
                augmented_strip = []

                for image in images_strip:
                    if len(image.shape) == 3:
                        image = np.expand_dims(image, axis=0)
                    else:
                        image = np.expand_dims(image, axis=0)
                        image = np.expand_dims(image, axis=3)

                    augmented_image_iterator = augment_processor.flow(image, batch_size=1)

                    for augmented_image in augmented_image_iterator:
                        image = np.squeeze(augmented_image, axis=0)
                        break
                    augmented_strip.append(image)

                print("Augmented strip number " + str(x + 1) + " generated")
                file_name = base_file_name + "Augmented" + str(x + 1) + "." + self.JPG_EXTENSION
                self.save_strip(augmented_strip, file_name)
            print("Saving original strip")
            file_name = base_file_name + "Original." + self.JPG_EXTENSION
            self.save_strip(images_strip, file_name)
        else:
            print("Not a valid image strip \n")

    @staticmethod
    def save_strip(strip, file_name):
        strip = util.combineImages(strip)
        cv2.imwrite(file_name, strip)

    def satisfy_frames_number(self, frames_array, required_length):
        if required_length > 0:
            if required_length > len(frames_array):
                self.increase_array_length(frames_array, required_length)
            elif required_length < len(frames_array):
                self.decrease_array_length(frames_array, required_length)
        return frames_array

    @staticmethod
    def increase_array_length(array, frames_number):
        difference = frames_number - len(array)
        last_index = len(array) - 1
        last_frame = array[last_index]
        for x in range(0, difference):
            array.append(last_frame)

    @staticmethod
    def decrease_array_length(array, frames_number):
        difference = len(array) - frames_number
        for x in range(0, difference):
            array.pop()

    def on_the_fly_calibration(self, sample_img, filter_index):
        if filter_index is self.BLURRY_EDGES or filter_index is self.EDGES:
            while True:
                kh = int(input("new kernel height: "))
                kw = int(input("new kernel width: "))
                if filter_index is self.EDGES:
                    new_img = self.edgeDetector.getImageEdgesFromNumpy(sample_img, kernelHeight=kh, kernelWidth=kw)
                else:
                    new_img = self.edgeDetector.getImageBluryEdgesFromNumpy(sample_img, kernelHeight=kh, kernelWidth=kw)
                cv2.imshow("Calibration", new_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                answer = input("Are you done with the values? (y/n): ")
                if answer is 'y':
                    break
            return kh, kw
        else:
            return 5, 5

    def apply_filter(self, images, filter_index, kh=5, kw=5):
        strip = []
        for image in images:
            if filter_index is self.NONE:
                strip.append(self.edgeDetector.make_square(image))
            elif filter_index is self.GRAY:
                strip.append(self.edgeDetector.toGrayscale(image))
            elif filter_index is self.BLURRY_EDGES:
                strip.append(self.edgeDetector.getImageBluryEdgesFromNumpy(image, kernelHeight=kh, kernelWidth=kw))
            else:
                strip.append(self.edgeDetector.getImageEdgesFromNumpy(image, kernelHeight=kh, kernelWidth=kw))
        return strip

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
