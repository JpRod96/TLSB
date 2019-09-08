import numpy as np
import cv2
from sklearn.metrics import pairwise

# Cargamos la imagen
original = cv2.imread("img.jpg")
# cv2.imshow("original", original)
 
# Convertimos a escala de grises
gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
cv2.imwrite ("gris.jpg", gris)

# Aplicar suavizado Gaussiano
blur = cv2.blur(gris, (5 ,5))
cv2.imwrite ("blur.jpg", blur)
# cv2.imshow("suavizado", gauss)


# Detectamos los bordes con Canny
canny = cv2.Canny(blur, 50, 150)
cv2.imwrite ("canny.jpg", canny)

 
newimg = cv2.resize(canny, (300,300))
cv2.imwrite ("resizeimg.jpg", newimg)

#cv2.imshow("canny", canny)

newimg = cv2.resize(canny, (300,300))
#cv2.imshow("newimg", newimg)
# print(canny.shape)
# rows,cols = canny.shape

# M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
# dst = cv2.warpAffine(canny,M,(cols,rows))
# print(dst.shape)
# cv2.waitKey(0)



## Buscamos los contornos
#(contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 
## Mostramos el número de monedas por consola
#print("He encontrado {} objetos".format(len(contornos)))
# 
#cv2.drawContours(original,contornos,-1,(0,0,255), 2)
#cv2.imshow("contornos", original)
 

#cap = cv2.VideoCapture(0)
#
#while True:
#    ret, frame = cap.read(0)
#
#    cv2.imshow('Video', frame)
#
#    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    med_val = np.median(gris)
#    lower = int(max(0,0.7*med_val))
#    upper = int(min(255,1.3*med_val))
#    blurred_img = cv2.blur(gris,(5,5))
#    canny = cv2.Canny(blurred_img, 50, 150)
#
#    cv2.imshow('Bordes', canny)
#
#    k = cv2.waitKey(1)
#
#    if k == 27:
#        break
#
#cap.release()
#cv2.detroyAllWindows()



#background = None
## Start with a halfway point between 0 and 1 of accumulated weight\n,
#accumulated_weight = 0.5
## Manually set up our ROI for grabbing the hand,
## Feel free to change these. I just chose the top right corner for filming,
#roi_top = 20
#roi_bottom = 450
#roi_right = 100
#roi_left = 550
#
#def calc_accum_avg(frame,accumulated_weight):
#
#    global background
#
#    if background is None:
#        background = frame.copy().astype('float')
#        return None
#
#    cv2.accumulateWeighted(frame,background,accumulated_weight)
#
#
#def segment(frame, threshold_min = 25):
#
#    diff = cv2.absdiff(background.astype('uint8'),frame)
#
#    ret, thresholded = cv2.threshold(diff,threshold_min,255,cv2.THRESH_BINARY)
#
#    contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#    if len(contours) == 0:
#        return None
#
#    else:
#        hands_segment = max(contours,key = cv2.contourArea)
#        
#        return(thresholded, hands_segment)
#
#
#def count_fingers(thresholded, hand_segment):
# # Calculated the convex hull of the hand segment,
# conv_hull = cv2.convexHull(hand_segment)
# # Now the convex hull will have at least 4 most outward points, on the top, bottom, left, and right.,
# # Let's grab those points by using argmin and argmax. Keep in mind, this would require reading the documentation,
# # And understanding the general array shape returned by the conv hull.,
# # Find the top, bottom, left , and right.,
# # Then make sure they are in tuple format,
# top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
# bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
# left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
# right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
# # In theory, the center of the hand is half way between the top and bottom and halfway between left and right,
# cX = (left[0] + right[0]) // 2
# cY = (top[1] + bottom[1]) // 2
# # find the maximum euclidean distance between the center of the palm,
# # and the most extreme points of the convex hull,
# # Calculate the Euclidean Distance between the center of the hand and the left, right, top, and bottom.,
# distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
# # Grab the largest distance,
# max_distance = distance.max()
# # Create a circle with 90% radius of the max euclidean distance,
# radius = int(0.8 * max_distance)
# circumference = (2 * np.pi * radius)
# # Not grab an ROI of only that circle,
# circular_roi = np.zeros(thresholded.shape[:2], dtype='uint8')
# # draw the circular ROI,
# cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
# # Using bit-wise AND with the cirle ROI as a mask.,
# # This then returns the cut out obtained using the mask on the thresholded hand image.,
# circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
# # Grab contours in circle ROI,
# contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# # Finger count starts at 0,
# count = 0
# # loop through the contours to see if we count any more fingers.,
# for cnt in contours:
#     # Bounding box of countour,
#     (x, y, w, h) = cv2.boundingRect(cnt)
#     # Increment count of fingers based on two conditions:,
#     # 1. Contour region is not the very bottom of hand area (the wrist),
#     out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
#     # 2. Number of points along the contour does not exceed 25% of the circumference of the circular ROI (otherwise we're counting points off the hand),
#     limit_points = ((circumference * 0.25) > cnt.shape[0])
#     if  out_of_wrist and limit_points:
#         count += 1
# return count
#
#
#
#cap = cv2.VideoCapture(0)
#
#num_frames = 0
#while True:
#    ret, frame = cap.read(0)
#
#    frame_copy = frame.copy()
#    
#    roi = frame[roi_top:roi_bottom,roi_right:roi_left]
#
#    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#
#    gray = cv2.GaussianBlur(gray,(7,7),0)
#
#    if num_frames < 60:
#        calc_accum_avg(gray,accumulated_weight)
#
#        if num_frames <= 59:
#            cv2.putText(frame_copy, 'WAIT GETTING BG',(200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#            cv2.imshow('Finger Count', frame_copy)
#    else:
#        hand = segment(gray)
#
#        if hand is not None:
#            thresholded, hand_segment = hand
#            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0),1)
#            fingers = count_fingers(thresholded, hand_segment)
#            cv2.putText(frame_copy, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#            cv2.imshow('Thesholded', thresholded)
#    
#    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)
#    num_frames += 1
#    cv2.imshow('Finger Count', frame_copy)
#
#    k = cv2.waitKey(1)
#
#    if k == 27:
#        break
#
#cap.release()
#cv2.detroyAllWindows()

# from imageai.Detection import ObjectDetection
# import os

# execution_path = os.getcwd()


# detector = ObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
# detector.loadModel()

# detections = detector.detectCustomObjectsFromImage(custom_objects=None, input_image=os.path.join(execution_path , "201.jpg"), output_image_path=os.path.join(execution_path , "image3custom.jpg"), minimum_percentage_probability=70)
# print(detections)


#original = cv2.imread("persona.jpg")
#print(new_model.predict_classes(original))