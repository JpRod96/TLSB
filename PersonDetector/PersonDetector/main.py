import numpy as np
import cv2
from PersonDetector import PersonDetector

PHOTO="photo"
JPG_EXTENSION=".jpg"
cap = cv2.VideoCapture('omg.mp4')
#numero total de frames
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cutOn = 0
cont = 0
listOfFrames = []

while(cap.isOpened()):
    ret, frame = cap.read()
    cont += 1
    if cont >= cutOn:
        listOfFrames.append(frame)
        # Corte en el frame:
        cutOn = frameCount/5 + cont
    else:
        if cutOn >= frameCount:
            break

numberOfFrame = 1
for photo in listOfFrames:
    imageName = PHOTO + str(numberOfFrame) + JPG_EXTENSION;
    cv2.imwrite (imageName, photo)
    numberOfFrame += 1

cap.release()
detector = PersonDetector();
listOfTreatedImages = []
for x in range(1, 6):
    imageName = PHOTO + str(x)
    print(imageName + JPG_EXTENSION)
    treatedImageName = detector.detectPerson(imageName, JPG_EXTENSION)
    listOfTreatedImages.append(treatedImageName)
