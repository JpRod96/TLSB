from imageai.Detection import ObjectDetection
import os
class PersonDetector:

    def detectPerson(self, imageName):
        execution_path = os.getcwd()
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
        detector.loadModel()

        custom_objects = detector.CustomObjects(person=True)
        detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , imageName), output_image_path=os.path.join(execution_path , imageName + "new.jpg"), minimum_percentage_probability=55)

        for detection in detections:
            print(detection["name"] , " : " , detection["percentage_probability"], ":", detection["box_points"])


        #detections, extracted_images = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), 
        #                                                               output_image_path=os.path.join(execution_path, "imagenew.jpg"), 
        #                                                               extract_detected_objects=True)
        