
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import base64
import logging
import time
import json

import sys
sys.path.insert(0, './helpers/')
from view import showImage

faceRegionIndices = {
    "rightBlush": [28, 29, 30, 3, 2, 1],
    "leftBlush": [28, 29, 30, 13, 14, 15],
    "rightEyeLeftMost": [39],
    "leftEyeRightMost": [42],
    "wholeFace": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 78, 74, 79, 73, 72, 80, 71, 70, 69, 63, 76, 75, 77]
}
tillShiftPoint = 3

colorRanges = {
    "lightPink": {
        # "lower": [0, 60, 193],
        # "higher": [11, 134, 209]
        "lower": [0, 42, 193],
        "higher": [12, 58, 255]
    },
    "darkPink": {
        "lower": [2, 109, 172],
        "higher": [9, 130, 239]
    },
    "lightRed": {
        "lower": [0, 79, 185],
        "higher": [9, 134, 255]
    },
    "darkRed": {
        "lower": [0, 79, 44],
        "higher": [5, 255, 255]
    }
}

def getPostShiftRegions(regions, cutLandmark, tillShiftPoint, shiftType):
    shiftCoordinate = 0 if shiftType == 'horizontal' else 1
    for item in regions[:tillShiftPoint]:
        item[shiftCoordinate] = cutLandmark[shiftCoordinate]
    return regions

def getColorFilteredImage(image, lower_range, upper_range, preview=False):
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_range, np.uint8) 
    upper = np.array(upper_range, np.uint8)
    mask = cv2.inRange(hsvFrame, lower, upper) 
    res = cv2.bitwise_and(image, image,  
                                  mask = mask) 
    if preview:
        showImage("color filtered", res)
    return res

def getLargestContourArea(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    if len(cnts) > 0:
        return cv2.contourArea(cnts[0])
    else:
        return 0

# loop over the face detections
def detectColoredBlush(image, imageName="test1.jpg", savePath='./extractedImages/', preview=False):
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    overlay = image.copy()
    output = image.copy()
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the face number
        frame = image.copy()
        cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (i, (x, y)) in enumerate(shape):
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
#             cv2.putText(frame, str(i + 1), (x - 10, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        if h > 0 and w > 0 and preview:
            showImage("Frame", frame[y:y+h, x:x+w])
        rightBlushPoints = shape[faceRegionIndices['rightBlush']]
        rightShiftLandmark = shape[faceRegionIndices['rightEyeLeftMost']]
#         print(f"right blush points: {rightBlushPoints}\n right shift coordinate: {rightShiftLandmark}")
        rightBlushPoints = getPostShiftRegions(rightBlushPoints, rightShiftLandmark[0], tillShiftPoint, 'horizontal')
#         print(f"right blush points: {rightBlushPoints}")

        leftBlushPoints = shape[faceRegionIndices['leftBlush']]
        leftShiftLandmark = shape[faceRegionIndices['leftEyeRightMost']]
        leftBlushPoints = getPostShiftRegions(leftBlushPoints, leftShiftLandmark[0], tillShiftPoint, 'horizontal')
        
        faceRegionROIs = {
            "rightBlush": rightBlushPoints,
            "leftBlush": leftBlushPoints
        }
        
        for regionName, regionIndices in faceRegionROIs.items():
            pts = regionIndices
            hull = cv2.convexHull(pts)
            overlayGray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            regionMask = np.zeros(overlayGray.shape, np.uint8)
            cv2.drawContours(regionMask, [hull], -1, (255, 255, 255), -1)
            roi = cv2.bitwise_and(overlay, overlay, mask = regionMask) 
            
            if preview:
                showImage(f"{regionName}", roi)
            grayROI = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(grayROI, 30, 200) 
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
            
            (x1, y1, w1, h1) = cv2.boundingRect(hull)
            if h1 > 0 and w1 > 0 and x1 > 0 and y1 > 0:
                # cv2.imwrite(savePath+regionName+"_"+imageName, roi[y1:y1+h1, x1:x1+w1])
                if preview:
                    showImage(f"Contours inside {regionName}", roi)
                blushColorPresence = {}
                for colorName in colorRanges.keys():
                    lowerColor = colorRanges[colorName]['lower']
                    higherColor = colorRanges[colorName]['higher']
                    if len(roi[y1:y1+h1, x1:x1+w1]) > 0:
                        try:
                            colorFilteredROI = getColorFilteredImage(roi[y1:y1+h1, x1:x1+w1], lowerColor, higherColor)
                        except Exception as e:
                            print(e)
                            raise Exception

                        colorDetectedContourArea = getLargestContourArea(colorFilteredROI)
                        roiContourArea = getLargestContourArea(roi[y1:y1+h1, x1:x1+w1])
                        # print(colorDetectedContourArea, roiContourArea)

                        colorAreaOverlappingRatio = 100*colorDetectedContourArea/roiContourArea
                        if colorAreaOverlappingRatio > 23:
        #                     print(f"Blush with specific {colorName} found")
                            blushColorPresence[colorName] = (blushColorPresence.get(colorName, colorAreaOverlappingRatio) + colorAreaOverlappingRatio)/2
            
        
        if preview:
            showImage(f"Rectangle: {i}", image[y:y+h, x:x+w])
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        # print(blushColorPresence)
        if len(blushColorPresence.keys()) == 0:
            print("No Blush found in face")
            return None
        blushColorPresence = dict(sorted(blushColorPresence.items(), key=lambda item: item[1], reverse=True))
        for k in blushColorPresence.items():
            print(f"Blush with color: {k} found")
            break
        
        return k


def init():
    logging.info('loading the model')
    try:
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        global detector
        global predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('./facial-landmarks/shape_predictor_81_face_landmarks.dat')
        print("models loaded successfully")
    except Exception as e:
        logging.error(e)

def run(raw_data):
    try:
        print('runmethod')
        start_time = time.time()
        image = json.loads(raw_data)["image"]
        encoded_img=bytes(image, encoding='utf8')
        image = np.array(Image.open(BytesIO(base64.b64decode(encoded_img))))
        blush = detectColoredBlush(image)
        time_taken = (time.time() - start_time)
        if blush != None:
            blushDetection = 1
        else:
            blushDetection = 0
        response = {"blush_detection": str(blushDetection) ,"Processing_time": str(time_taken)}
        return response
    except IndexError as indexError:
        print(indexError)
        return "error detecting blush"
