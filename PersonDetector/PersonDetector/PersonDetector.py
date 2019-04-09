from imageai.Detection import ObjectDetection
import os
import numpy as np
import cv2

class PersonDetector:

    detector=None
    execution_path=None

    def __init__(self): 
        self.execution_path = os.getcwd()
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath( os.path.join(self.execution_path , "resnet50_coco_best_v2.0.1.h5"))
        self.detector.loadModel()

    def detectPerson(self, imageName, extension):
        custom_objects = self.detector.CustomObjects(person=True)
        detections = self.detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
                                                               input_image=os.path.join(self.execution_path , imageName + extension),
                                                               output_image_path=os.path.join(self.execution_path , imageName + "new" + extension),
                                                               minimum_percentage_probability=55)
        highestPercentageDetection = self.getDetectionWHighestPercentage(detections)
        return self.cropImage(highestPercentageDetection["box_points"], imageName, extension)

    def cropImage(self, array, imageName, extension):
        image = cv2.imread(imageName + extension)
        x0=self.cleanNumber(array[0])
        y0=self.cleanNumber(array[1])
        x1=self.cleanNumber(array[2])
        y1=self.cleanNumber(array[3])
        cropped = image[y0:y1, x0:x1]
        treatedImageName=imageName + "PD"
        cv2.imwrite(treatedImageName + extension, cropped)  
        return treatedImageName

    def cleanNumber(self, number):
        return number if number>0 else 1

    def getDetectionWHighestPercentage(self, detections):
        highest = detections[0]
        for detection in detections:
            highest = detection if detection["percentage_probability"] > highest["percentage_probability"] else highest
        return highest
    