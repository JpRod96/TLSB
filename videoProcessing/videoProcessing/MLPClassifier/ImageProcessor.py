import numpy as np
import cv2


class ImageProcessor:

    BLURRY_EDGES = 1
    EDGES = 2

    def __init__(self, image_final_size, crop_size, kernel_height=5, kernel_width=5, process_type=EDGES):
        self.image_final_size = image_final_size
        self.image_crop_size = crop_size
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.process_type = process_type

    def process(self, image):
        if self.image_final_size > 0:
            image = self.rescale(image)
        if self.process_type is self.BLURRY_EDGES:
            return self.get_image_blurry_edges(image)
        elif self.process_type is self.EDGES:
            return self.get_image_edges(image)
        else:
            return image

    def rescale(self, image):
        return cv2.resize(image, (self.image_final_size, self.image_final_size))

    def get_image_blurry_edges(self, image):
        med_val = np.median(image)
        lower = int(max(0, 0.7 * med_val))
        upper = int(min(255, 1.3 * med_val))
        blurred_img = cv2.blur(image, (self.kernel_height, self.kernel_width))
        canny = cv2.Canny(blurred_img, lower, upper)
        return canny

    def get_image_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray, (self.kernel_height, self.kernel_width), 0)
        canny = cv2.Canny(gauss, 30, 90)
        return canny
