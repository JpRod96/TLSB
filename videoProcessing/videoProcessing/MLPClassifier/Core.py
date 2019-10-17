from videoCapture import VideoCapture


class Core:
    FROM_PATH = 1
    FROM_CAM = 2

    def __init__(self, source_type=FROM_CAM, path=None, image_processor=None, classifier=None):
        self.video_capture = VideoCapture()
        self.source_type = source_type
        self.path = path
        self.image_processor = image_processor
        self.classifier = classifier

    def start(self):
        if self.source_type is self.FROM_CAM:
            self.video_capture.capture_from_camera(processor=self.image_processor, classifier=self.classifier)
        elif self.source_type is self.FROM_PATH:
            self.video_capture.capture_from_file(self.path, processor=self.image_processor, classifier=self.classifier)
        else:
            print("Not valid a option")
