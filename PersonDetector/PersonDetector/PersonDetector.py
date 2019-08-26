from imageai.Detection import ObjectDetection
import os
import numpy as np
import cv2

class PersonDetector:

    detector=None
    execution_path=None
    custom_objects=None
    SLACKED_PIXELS = 40

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
    
    def detectPersonFromNumpy(self, image):
        npArray, detections = self.detector.detectCustomObjectsFromImage(custom_objects=self.custom_objects,
                                                                input_type="array",
                                                                input_image=image,
                                                                output_type="array",
                                                                minimum_percentage_probability=55)
        highestPercentageDetection = self.getDetectionWHighestPercentage(detections)
        if(highestPercentageDetection != None):
            return self.cropImage(highestPercentageDetection["box_points"], image)
        else:
            raise Exception('No human were found')

    def cropImage(self, array, image):
        print(image.shape)
        im_height, im_width, channels = image.shape
        x0=self.treatNumberLower(array[0])
        y0=self.treatNumberLower(array[1])
        x1=self.treatNumberUpper(array[2], im_width)
        y1=self.treatNumberUpper(array[3], im_height)
        return image[y0:y1, x0:x1]

    def treatNumberLower(self, number):
        slackedNumber = number - self.SLACKED_PIXELS
        return slackedNumber if slackedNumber>0 else 1
    
    def treatNumberUpper(self, number, maxNumber):
        slackedNumber = number + self.SLACKED_PIXELS
        return slackedNumber if slackedNumber <= maxNumber else number

    def getDetectionWHighestPercentage(self, detections):
        highest = None
        if(len(detections)>0):
            highest = detections[0]
            for detection in detections:
                highest = detection if detection["percentage_probability"] > highest["percentage_probability"] else highest
        return highest
    