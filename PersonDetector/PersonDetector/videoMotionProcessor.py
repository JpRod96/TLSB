from videoProcessorI import VideoProcessorI
# import matplotlib.pyplot as plt
import util
import cv2
import numpy as np
from scipy.signal import argrelmax
from scipy.signal import argrelmin
from scipy import ndimage
import os
from edgeDetector import EdgeDetector
from PersonDetector import PersonDetector


class VideoMotionProcessor(VideoProcessorI):
    iterations = 4
    MAX_ALIKE_PERCENTAGE = 31
    NONE = 'None'
    EDGE = 'Edges'
    BLURRY_EDGE = 'Blurry_Edges'
    GRAYSCALE = 'Grayscale'
    AUGMENTED_SIZE = 5

    def __init__(self, final_pic_size, to_combine, frames_nr=0, rotate=False, img_filter=NONE, aug_processor=None):
        self.picSize = final_pic_size
        self.edgeDetector = EdgeDetector(final_pic_size)
        self.detector = PersonDetector()
        self.combineImages = to_combine
        self.framesNumber = frames_nr
        self.rotateImages = rotate
        self.imageFilter = img_filter
        self.augmentedProcessor = aug_processor
        self.videoName = ""
        self.directory = ""

    def process(self, video_path):
        frames = self.cutVideo(video_path)
        record = self.get_alike_weights(frames)
        self.directory = util.getPathOfVideoDirectory(video_path)
        self.videoName = util.getLastTokenOfPath(video_path)[0]

        maxima_indexes, maxima_values, global_indexes = self.process_for_getting_max_alike(record)

        print(maxima_indexes)
        print(maxima_values)
        print(global_indexes)

        global_indexes = self.post_process_filter(frames, global_indexes)

        if self.combineImages:
            self.combine_frames(frames, global_indexes)
        else:
            self.save_critical_frames(frames, global_indexes)

    def post_process_filter(self, critical_frames, indexes):
        print("Starting post-process filter")

        final_frames_indexes = []
        compare_frame_index = indexes[0]
        compare_frame = critical_frames[compare_frame_index]
        cont = 0

        for index in range(0, len(indexes) - 1):
            global_index = indexes[index + 1]
            actual_frame = critical_frames[global_index]

            match = self.compare(compare_frame, actual_frame)
            # print(str(match))

            if match < self.MAX_ALIKE_PERCENTAGE:
                final_frames_indexes.append(compare_frame_index)
            compare_frame = actual_frame
            compare_frame_index = global_index
            cont += 1
            self.print_progress_bar(cont, len(indexes) - 1, prefix='Post-Process:', suffix='Complete', length=25)

        final_frames_indexes.append(compare_frame_index)
        return final_frames_indexes

    def process_for_getting_max_alike(self, weigths):
        maxima_values = weigths
        maxima_indexes = None
        global_indexes = np.arange(0, len(maxima_values), 1)

        # plt.plot(maximaValues)
        # plt.ylabel('Alike percentage')
        # plt.show()

        for x in range(1, self.iterations):
            maxima_indexes = argrelmax(maxima_values)[0]
            maxima_values = maxima_values[maxima_indexes]
            global_indexes = global_indexes[maxima_indexes]

        return maxima_indexes, maxima_values, global_indexes

    def process_for_getting_min_alike(self, weigths):
        minima_values = weigths
        minima_indexes = None
        global_indexes = np.arange(0, len(minima_values), 1)

        # plt.plot(minimaValues)
        # plt.ylabel('Alike percentage')
        # plt.show()

        for x in range(1, self.iterations):
            minima_indexes = argrelmin(minima_values)[0]
            minima_values = minima_values[minima_indexes]
            global_indexes = global_indexes[minima_indexes]

        return minima_indexes, minima_values, global_indexes

    def save_critical_frames(self, frames, indexes):
        counter = 1
        new_path = self.directory + "/" + self.videoName
        os.mkdir(new_path)
        for index in indexes:
            print("Processing frame number " + str(counter) + "...")
            file_name = new_path + "/" + self.videoName + str(counter)
            counter += 1
            frame = frames[index]
            if self.rotateImages:
                frame = ndimage.rotate(frame, 270)
            try:
                frame = self.detector.detectPersonFromNumpy(frame)
                if self.picSize > 0:
                    frame = self.apply_filter(frame)
                print("Done.\n")
                cv2.imwrite(file_name + ".jpg", frame)
            except:
                print("Human not found on frame number " + str(counter - 1))

    def combine_frames(self, frames, indexes):
        x = 1
        edge_images = []
        for index in indexes:
            frame = frames[index]
            if self.rotateImages:
                frame = ndimage.rotate(frame, 270)
            print("Processing frame number " + str(x) + "...")
            try:
                treated_image = self.detector.detectPersonFromNumpy(frame)
                edge_images.append(self.apply_filter(treated_image))
                print("Done.\n")
            except:
                print("Human not found on frame number " + str(x))
            x = x + 1
        edge_images = self.satisfy_frames_number(edge_images)
        file_name = self.videoName + "Edges"
        self.concat_images(edge_images, file_name)

    def concat_images(self, images, file_name):
        if self.augmentedProcessor is None:
            if len(images) > 0:
                self.create_strip(images, file_name)
            else:
                print("Unsuccessful process, there's no human on the video clip " + self.videoName + "\n")
        else:
            self.augment_image_strip(images)

    def create_strip(self, image_strip, file_name):
        print("Concatenating all images")
        data = util.combineImages(image_strip)
        util.saveImageToPath(data, file_name, ".jpg", self.directory)
        print("Done.")

    def augment_image_strip(self, images_strip):
        new_directory = self.videoName + "Augmented"
        os.mkdir(self.directory + "/" + new_directory)
        if len(images_strip) > 0:
            print("Augmenting image strip...")

            for x in range(0, self.AUGMENTED_SIZE):
                augmented_strip = []

                for image in images_strip:
                    if len(image.shape) == 3:
                        image = np.expand_dims(image, axis=0)
                    else:
                        image = np.expand_dims(image, axis=0)
                        image = np.expand_dims(image, axis=3)

                    augmented_image_iterator = self.augmentedProcessor.flow(image, batch_size=1)

                    for augmented_image in augmented_image_iterator:
                        image = np.squeeze(augmented_image, axis=0)
                        break
                    augmented_strip.append(image)

                print("Augmented strip number " + str(x + 1) + " generated")
                file_name = new_directory + "/" + self.videoName + "EdgesAugmented" + str(x + 1)
                self.create_strip(augmented_strip, file_name)
            print("Saving original strip")
            file_name = new_directory + "/" + self.videoName + "EdgesOriginal"
            self.create_strip(images_strip, file_name)
        else:
            print("Unsuccessful process, there's no human on the video clip " + self.videoName + "\n")

    def apply_filter(self, frame):
        filter_name, kh, kw = self.get_filter_token(self.imageFilter)
        if filter_name == self.EDGE:
            return self.edgeDetector.getImageEdgesFromNumpy(frame, kernelHeight=kh, kernelWidth=kw)
        elif filter_name == self.GRAYSCALE:
            return self.edgeDetector.toGrayscale(frame)
        elif filter_name == self.BLURRY_EDGE:
            return self.edgeDetector.getImageBluryEdgesFromNumpy(frame, kernelHeight=kh, kernelWidth=kw)
        else:
            square_pic = self.edgeDetector.make_square(frame)
            return cv2.resize(square_pic, (self.picSize, self.picSize))

    def get_filter_token(self, string):
        tokens = string.split()
        if len(tokens) >= 3:
            return tokens[0], int(tokens[1]), int(tokens[2])
        else:
            return string, 5, 5

    def satisfy_frames_number(self, frames_array):
        if self.framesNumber > 0:
            if self.framesNumber > len(frames_array):
                self.increaseArrayLength(frames_array)
            elif self.framesNumber < len(frames_array):
                self.decreaseArrayLength(frames_array)
        return frames_array

    def increaseArrayLength(self, array):
        difference = self.framesNumber - len(array)
        lastIndex = len(array) - 1
        lastFrame = array[lastIndex]
        for x in range(0, difference):
            array.append(lastFrame)

    def decreaseArrayLength(self, array):
        difference = len(array) - self.framesNumber
        for x in range(0, difference):
            array.pop()

    def cutVideo(self, videoPath):
        cap = cv2.VideoCapture(videoPath)
        print("Video " + videoPath + " loaded")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(str(frame_count) + " frames to process...")
        list_of_frames = []
        cont = 0
        while cap.isOpened() and cont < frame_count:
            ret, frame = cap.read()
            list_of_frames.append(frame)
            cont += 1
        cap.release()
        return list_of_frames

    def get_alike_weights(self, frames):
        frame_count = len(frames)
        weights = []
        cont = 0
        last_frame = None
        on_going_frame = None

        for index in range(0, frame_count):
            frame = frames[index]
            last_frame = on_going_frame
            on_going_frame = frame
            cont += 1
            if cont > 1:
                match = self.compare(last_frame, on_going_frame)
                weights.append(match)
                self.print_progress_bar(cont, frame_count, prefix='Progress:', suffix='Complete', length=50)
            if cont >= frame_count:
                break
        return np.array(weights)

    def compare(self, pic1, pic2):
        sift = cv2.xfeatures2d.SIFT_create()
        kp_1, desc_1 = sift.detectAndCompute(pic1, None)
        kp_2, desc_2 = sift.detectAndCompute(pic2, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        matchPercentage = len(good_points) / number_keypoints * 100
        return matchPercentage

    # make utilitary

    def print_progress_bar(self, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
        if iteration == total:
            print()
