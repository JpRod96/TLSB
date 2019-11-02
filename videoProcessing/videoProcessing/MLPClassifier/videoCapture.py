import cv2


class VideoCapture:

    def capture_from_file(self, path, processor=None, classifier=None):
        evolution = []
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            frame = cv2.resize(frame, (960, 540))
            if processor is not None:
                treated = processor.process(frame)
            else:
                treated = frame
            if classifier is not None:
                prediction = classifier.predict(treated)
                if prediction is not None:
                    evolution.append(prediction)
            cv2.imshow('frame', treated)
            cv2.imshow('not', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return evolution

    def capture_from_camera(self, processor=None, classifier=None):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if processor is not None:
                treated = processor.process(frame)
            else:
                treated = frame
            if classifier is not None:
                classifier.predict(treated)
            cv2.imshow('frame', treated)
            cv2.imshow('not', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.waitKey(33) == ord('+'):
                processor.kernel_width = processor.kernel_width + 1
                processor.kernel_height = processor.kernel_height + 1

            if cv2.waitKey(33) == ord('-'):
                processor.kernel_width = processor.kernel_width - 1
                processor.kernel_height = processor.kernel_height - 1
        cap.release()
        cv2.destroyAllWindows()
