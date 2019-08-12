#code to only detect person class

import cv2
import time
import imutils
from imutils.video import VideoStream
import numpy as np

#args to load the model

args = {
    "prototxt":"mobilenet_ssd/MobileNetSSD_deploy.prototxt",
    "model":"mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
    "skip-frames":25
}

#classes that mobilenet was trained on .
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("Videostream started")
# vs = VideoStream(0).start()

vs = cv2.VideoCapture('compress.mp4')
# time.sleep(1.0)

Height=None
Width=None

while True:

    ret,frame = vs.read()
    if frame is None:
        break
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = imutils.resize(frame, width=400)
    # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    (Height , Width) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (Width, Height), 127.5)

    net.setInput(blob)

    detections = net.forward()

    # print(detections.shape[2])

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            idx = int(detections[0, 0, i, 1])  # extract the index of the class label from the `detections'

            if CLASSES[idx] != "person":
                continue

            box = detections[0, 0, i, 3:7] * np.array([Width,Height,Width,Height])
            print(box)
            (startX, startY, endX, endY) = box.astype("int")

            #for displaying

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print(CLASSES[idx])
            print(confidence *100)

            cv2.rectangle(frame, (startX, startY), (endX, endY),(0,0,255), 2)

            y = startY - 15 if startY - 15 > 15 else startY + 15

            cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

            
vs.release()

# close any open windows
cv2.destroyAllWindows()


