from Core import Core
from ImageProcessor import ImageProcessor
from SignClassifier import MLPSignClassifier
import sys
sys.path.insert(0, 'D:/desktop/TLSB/PersonDetector/PersonDetector')
from HaarCascadeProcessor import HaarCascadeProcessor

haar = HaarCascadeProcessor()
classifier_path = "D:/desktop/TLSB/FirstPyNN/FirstPyNN/MLP/bestModel80.h5"
#path = "D:/desktop/DATASET/ADIOS/20190812_160131.mp4"
path = "D:/desktop/DATASET/CBBA/20190908_185548.mp4"
classifier = MLPSignClassifier(classifier_path)
proc = ImageProcessor(300, 0, process_type=ImageProcessor.BLURRY_EDGES, kernel_height=5, kernel_width=5)
core = Core(Core.FROM_CAM, path=path, image_processor=proc, classifier=classifier)

core.start()
#core.capture_from_camera(haar)
