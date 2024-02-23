import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import arabic_reshaper
import bidi.algorithm



cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model6/keras_model.h5","Model6/labels.txt")

offset = 20
imgsize = 300
##folder = "IMAGE/3"
counter = 0
labels =[u"اهلا","i Love You"]
while True:
    success, img = cap.read()
    imgoutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
        imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgcropShape = imgcrop.shape
        aspecratio = h / w
        if aspecratio > 1:
            k = imgsize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgcrop,(wCal, imgsize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgsize - wCal) / 2)
            imgwhite[:, wGap:wCal + wGap] = imgResize
            prediction, index =  classifier.getPrediction(imgwhite, draw=False)
            print(prediction, index)

        else:
            k = imgsize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgcrop,(imgsize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgsize - hCal) / 2)
            imgwhite[hGap:hCal + hGap, :] = imgResize
            imgwhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgwhite, draw=False)
            cv2.rectangle(imgoutput, (x - offset, y - offset-50),
                          (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED),(x - offset, y - offset-50)
        cv2.putText(imgoutput, labels[index],(x,y-26),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1.7,(255,255,255,255),2)
        cv2.rectangle(imgoutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)


        cv2.imshow("imagecrop", imgcrop)
        cv2.imshow("imgwhite", imgwhite)

    cv2.imshow("IMAGE", imgoutput)
    cv2.waitKey(1)