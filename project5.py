import cv2 as cv
import HandTrackingModule as htm
import os
import numpy as np

cap = cv.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

folderPath = "Header"
myList = os.listdir(folderPath)
overLayList = []

for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overLayList.append(image)

print("Header Images Loaded:", len(overLayList))

header = overLayList[0]
detector = htm.handDetector(detectionCon=0.8)

color = (0, 255, 255)
brushThickness = 15
eraserThickness = 30

xp, yp = 0, 0
imgCanvas = np.zeros((480, 640, 3), np.uint8)

while True:
    ret, img = cap.read()
    img = cv.flip(img, 1)

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Get tip of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        # Selection mode (2 fingers up)
        if fingers[1] == 1 and fingers[2] == 1:
            xp, yp = 0, 0  # Reset previous point
            print("Selection mode")
            if y1 < 101:
                if 0 < x1 < 100:
                    header = overLayList[0]
                    color = (0, 255, 255)
                elif 100 < x1 < 200:
                    header = overLayList[1]
                    color = (0, 0, 255)
                elif 200 < x1 < 300:
                    header = overLayList[2]
                    color = (255, 0, 255)
                elif 300 < x1 < 400:
                    header = overLayList[3]
                    color = (255, 0, 0)
                elif 400 < x1 < 500:
                    header = overLayList[4]
                    color = (255, 255, 255)
                elif x1 > 500:
                    header = overLayList[5]
                    color = (0, 0, 0)  # eraser

            cv.rectangle(img, (x1 - 20, y1 - 35), (x2 + 20, y2 + 20), color, -1)

        # Drawing mode (only index finger up)
        elif fingers[1] == 1 and fingers[2] == 0:
            print("Drawing mode")
            cv.circle(img, (x1, y1), 10, color, -1)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if color == (0,0,0):
                # Eraser
                cv.line(img, (xp, yp), (x1, y1), color, eraserThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), color, eraserThickness)
            else:
                cv.line(img, (xp, yp), (x1, y1), color, brushThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), color, brushThickness)

            xp, yp = x1, y1

    # Merge canvas and video feed
    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)

    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, imgCanvas)

    # Resize header to match video width
    h, w, _ = img.shape
    header = cv.resize(header, (w, 101))
    img[0:101, 0:w] = header

    cv.imshow('Virtual Painter', img)
    # cv.imshow('Canvas', imgCanvas)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
