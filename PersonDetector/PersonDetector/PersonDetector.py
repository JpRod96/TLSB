from imageai.Detection import ObjectDetection
import os
import numpy as np
import cv2

class PersonDetector:

    detector=None
    execution_path=None
    custom_objects=None

    def __init__(self): 
        self.execution_path = os.getcwd()
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath( os.path.join(self.execution_path , "resnet50_coco_best_v2.0.1.h5"))
        self.detector.loadModel()
        self.custom_objects = self.detector.CustomObjects(person=True)

    def detectPerson(self, imageName, extension):
        image = cv2.imread(imageName + extension)
        detections = self.detector.detectCustomObjectsFromImage(custom_objects=self.custom_objects,
                                                               input_image=os.path.join(self.execution_path , imageName + extension),
                                                               output_image_path=os.path.join(self.execution_path , imageName + "new" + extension),
                                                               minimum_percentage_probability=55)
        highestPercentageDetection = self.getDetectionWHighestPercentage(detections)
        cropped = self.cropImage(highestPercentageDetection["box_points"], image)
        treatedImageName=imageName + "PD"
        cv2.imwrite(treatedImageName + extension, cropped)  
        return treatedImageName
    
    def detectPersonFronNumpy(self, imageName, extension):
        image = cv2.imread(imageName + extension)
        detections = self.detector.detectCustomObjectsFromImage(custom_objects=self.custom_objects,
                                                               input_image=os.path.join(self.execution_path , imageName + extension),
                                                               output_image_path=os.path.join(self.execution_path , imageName + "new" + extension),
                                                               minimum_percentage_probability=55)
        highestPercentageDetection = self.getDetectionWHighestPercentage(detections)
        return self.cropImage(highestPercentageDetection["box_points"], image)

    def cropImage(self, array, image):
        x0=self.cleanNumber(array[0])
        y0=self.cleanNumber(array[1])
        x1=self.cleanNumber(array[2])
        y1=self.cleanNumber(array[3])
        return image[y0:y1, x0:x1]

    def cleanNumber(self, number):
        return number if number>0 else 1

    def getDetectionWHighestPercentage(self, detections):
        highest = detections[0]
        for detection in detections:
            highest = detection if detection["percentage_probability"] > highest["percentage_probability"] else highest
        return highest
    