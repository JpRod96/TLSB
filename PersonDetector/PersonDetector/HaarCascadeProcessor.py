import cv2
from os import listdir
from os.path import isfile, join
import util


class HaarCascadeProcessor:

    def process_from(self, dir_path):
        files = self.get_image_files_from_directory(dir_path)
        for file in files:
            HaarCascadeProcessor.process(dir_path + "/" + file)

    @staticmethod
    def get_image_files_from_directory(path):
        return [f for f in listdir(path) if isfile(join(path, f))]

    @staticmethod
    def process(path):
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detector = cv2.CascadeClassifier("brazo_cascade.xml")
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
        print(rects)
        for (i, (x, y, w, h)) in enumerate(rects):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        file_name = util.getLastTokenOfPath(path)[0]
        util.saveImageToPath(image, file_name + "cascade", ".jpg", util.getPathOfVideoDirectory(path))
