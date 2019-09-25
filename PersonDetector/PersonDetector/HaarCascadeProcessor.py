import cv2


class HaarCascadeProcessor:

    @staticmethod
    def process(path):
        # load the input image and convert it to grayscale
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # load the cat detector Haar cascade, then detect cat faces
        # in the input image
        detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
        rects = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(120, 120))
        print(rects)
        for (i, (x, y, w, h)) in enumerate(rects):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # show the detected cat faces
        cv2.imshow("Cat Faces", image)
        cv2.waitKey(0)
