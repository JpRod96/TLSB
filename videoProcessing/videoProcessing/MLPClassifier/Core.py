from videoCapture import VideoCapture
import matplotlib.pyplot as plt


class Core:
    FROM_PATH = 1
    FROM_CAM = 2

    def __init__(self, source_type=FROM_CAM, path=None, image_processor=None, classifier=None, debug=False):
        self.video_capture = VideoCapture()
        self.source_type = source_type
        self.path = path
        self.image_processor = image_processor
        self.classifier = classifier
        self.debug = debug

    def start(self):
        if self.source_type is self.FROM_CAM:
            self.video_capture.capture_from_camera(processor=self.image_processor, classifier=self.classifier)
        elif self.source_type is self.FROM_PATH:
            array = self.video_capture.capture_from_file(self.path, processor=self.image_processor,
                                                         classifier=self.classifier)
            if self.debug:
                self.plot(array)
        else:
            print("Not valid a option")

    def plot(self, array):
        fig = plt.figure(figsize=(5, 5))

        plt.plot(array)
        plt.ylabel('classes')
        plt.xlabel('prediction')
        plt.grid(b=True)
        plt.suptitle('Evolution')
        plt.savefig("evolution" + '.png')
        plt.close(fig)
