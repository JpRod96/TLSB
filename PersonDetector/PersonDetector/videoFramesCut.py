import numpy as np
import cv2

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
        #print(cont)
    else:
        if cutOn >= frameCount:
            break

numberOfFrame = 1
for photo in listOfFrames:
    cv2.imshow('photo',photo)
    cv2.imwrite ("photo" + str(numberOfFrame) + ".jpg", photo)
    cv2.waitKey(0)
    numberOfFrame += 1

cap.release()
cv2.destroyAllWindows()