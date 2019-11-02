from Core import Core
from ImageProcessor import ImageProcessor
from SignClassifier import MLPSignClassifier
import sys
sys.path.insert(0, 'D:/desktop/TLSB/PersonDetector/PersonDetector')
from HaarCascadeProcessor import HaarCascadeProcessor

haar = HaarCascadeProcessor()
classifier_path = "D:/desktop/TLSB/FirstPyNN/FirstPyNN/MLP/bestModel80.h5"
raro = "D://desktop//TLSB//FirstPyNN//FirstPyNN//MLP//results//primer intento/bestModel80.h5"

POR_FAVOR = "D:/desktop/DATASET/2/POR_FAVOR/20191018_105014.mp4"
CAFE = "D:/desktop/DATASET/2/CAFE/20191018_104918.mp4"
HOLA = "D:/desktop/DATASET/2/HOLA/20191018_104850.mp4"
QUERER = "D:/desktop/DATASET/2/QUERER/20191018_105443.mp4"

CBBA = "D:/desktop/DATASET/2/CBBA/20191018_105049.mp4"
classifier = MLPSignClassifier(classifier_path)
proc = ImageProcessor(300, 0, process_type=ImageProcessor.BLURRY_EDGES, kernel_height=3, kernel_width=3)
core = Core(Core.FROM_PATH, path=CAFE, image_processor=proc, classifier=classifier, debug=True)

core.start()
#core.capture_from_camera(haar)
