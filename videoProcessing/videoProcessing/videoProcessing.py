import numpy as np
import cv2
from sklearn.metrics import pairwise
from keras.models import load_model
from keras.preprocessing import image

model = load_model('gestos2.h5')
background = None
accumulated_weight = 0.5
roi_top = 30
roi_bottom = 470
roi_right = 100
roi_left = 550

def calc_accum_avg(frame,accumulated_weight):

    global background

    if background is None:
        background = frame.copy().astype('float')
        return None

    cv2.accumulateWeighted(frame,background,accumulated_weight)

def segment(frame, threshold_min = 25):

    diff = cv2.absdiff(background.astype('uint8'),frame)

    ret, thresholded = cv2.threshold(diff,threshold_min,255,cv2.THRESH_BINARY)
        
    return thresholded


def recognition(thresholded):

    original = image.load_img("predc.jpg",target_size=(256,256))

    original = image.img_to_array(original)

    original = np.expand_dims(original, axis = 0)

    original = original / 255

#   print(max(max(model.predict(original))))

    return (model.predict_classes(original) , max(max(model.predict(original))))

def getWord(word):
    switcher = {
        0: "HOLA",
        1: "LA",
        2: "YO",
        3: "QUERER",
        4: "RER",
        5: "CAFE",
    }
    return switcher.get(word)


#PROGRAMA PRINCIPAL

cap = cv2.VideoCapture(0)
num_frames = 0
predictedGesturesList = []

while True:
    ret, frame = cap.read(0)
    frame_copy = frame.copy()    
    roi = frame[roi_top:roi_bottom,roi_right:roi_left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(1,3), 0)
    # OBTENER FONDO
    #if num_frames < 60:
    #    calc_accum_avg(gray,accumulated_weight)
    #    if num_frames <= 59:
    #        cv2.putText(frame_copy, 'WAIT GETTING BG',(200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    #        cv2.imshow('recognition', frame_copy)

    if num_frames < 60:
        if num_frames <= 59:
            cv2.putText(frame_copy, 'WAIT GETTING BG',(200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow('recognition', frame_copy)
    else:
        #else:
            # BORRAR FONDO
            #hand = segment(gray)
            #if hand is not None:
               #thresholded = hand
        med_val = np.median(gray)
        lower = int(max(0,0.7*med_val))
        upper = int(min(255,1.3*med_val))
        canny = cv2.Canny(gray, lower,upper)
        cv2.imwrite("predc.jpg", canny)
        prediction, probability = recognition(canny)

        if probability > 0.70:
            print(probability)
            word = getWord(prediction[0])
            cv2.putText(frame_copy, word, (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
            cv2.imshow('Thesholded', canny)
            predictedGesturesList.append(word)
            #print(predictedGesturesList)
        else:
            cv2.putText(frame_copy, 'None' , (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
            cv2.imshow('Thesholded', canny)

    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 3)
    num_frames += 1
    cv2.imshow('recognition', frame_copy)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
finalPredictedGesturesList = []
for i in predictedGesturesList:
    if i not in finalPredictedGesturesList:
        finalPredictedGesturesList.append(i)

print(finalPredictedGesturesList)
