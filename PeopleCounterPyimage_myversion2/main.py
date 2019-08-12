#This code will predict all the classes to which the mobilenet ssd was trained for

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

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("Videostream started")
vs = VideoStream(0).start()
time.sleep(1.0)

H=None
W=None

while True:

    frame = vs.read()
    if frame is None:
        break
    
    frame = imutils.resize(frame, width=500)

    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

    net.setInput(blob)

    detections = net.forward()

    # print(detections.shape[2])

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            idx = int(detections[0, 0, i, 1])  # extract the index of the class label from the `detections'
            print(idx)
            box = detections[0, 0, i, 3:7] * np.array([W,H,W,H])
            (startX, startY, endX, endY) = box.astype("int")

            #for displaying

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)

            print("{}".format(label))

            cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)

            y = startY - 15 if startY - 15 > 15 else startY + 15

            cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

            
vs.stop()


# close any open windows
cv2.destroyAllWindows()


