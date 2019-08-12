# import the necessary packages
from utils.centroidTracker import centroidTracker
from utils.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils		
import time
import dlib
import cv2

args = {   #also a confidence to filter out weak detections
	"prototxt":"mobilenet_ssd/MobileNetSSD_deploy.prototxt",
	"model":"mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
	"skip-frames":25
	}
# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("loading the model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if a video path was not supplied, grab a reference to the webcam

print("starting video stream...")
# vs = VideoStream('http://192.168.43.172:4747/mjpegfeed').start()
# vs = VideoStream(0).start()
# time.sleep(1.0)
vs = cv2.VideoCapture('compress.mp4')
W = None
H = None
# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = centroidTracker(maxDisappeared=80, maxDistance=20)    #CT
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	ret, frame = vs.read()
	# frame = vs.read()
	if frame is None:
		break
	# cv2.imshow('frame',frame)

	frame = imutils.resize(frame, width=600)  #for the processing
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB for dlib, Since DLIB requires the image in RGB

	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	status = "RUNNING"
	rects = []   #the box obtained over a [person]  (populated either via detection or tracking)

	# check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
	if totalFrames % 30 == 0:
		# set the status and initialize our new set of object trackers
		status = "DETECTING"
		trackers = []

		# convert the frame to a blob and pass the blob through the
		# network and obtain the detections
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		# blob = cv.dnn.blobFromImage(image, scalefactor, size, mean, swapRB, crop)
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum
			# confidence
			if confidence > 0.6:
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1])   #gives the calss label

				# if the class label is not a person, ignore it
				if CLASSES[idx] != "person":
					continue

				# compute the (x, y)-coordinates of the bounding box
				# for the object
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])

				(startX, startY, endX, endY) = box.astype("int")

				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
				tracker.start_track(rgb, rect)

				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				trackers.append(tracker)

	# otherwise, we should utilize our object *trackers* rather than
	# object *detectors* to obtain a higher frame processing throughput
	else:
		# loop over the trackers
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "TRACKING"

			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))

	# draw a horizontal line in the center of the frame -- once an
	# object crosses this line we will determine whether they were
	# moving 'up' or 'down'
	cv2.line(frame, (0, H // 2), (W, H // 2), (0, 200, 155), 1)

	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects = ct.update(rects)
	print(objects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		print(objectID)
		print(type(centroid))
		print(centroid)

		# check to see if a trackable object exists for the current
		# object ID
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

		# otherwise, there is a trackable object so we can utilize it
		# to determine direction
		else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
			y = [c[1] for c in to.centroids]
			print(y)
			direction = centroid[1] - np.mean(y)
			print(direction)
			to.centroids.append(centroid)

			# check to see if the object has been counted or not
			if not to.counted:
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line, count the object
				if direction < 0 and centroid[1] < H // 2:
					totalUp += 1
					to.counted = True

				# if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, count the object
				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					to.counted = True

		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# construct a tuple of information we will be displaying on the
	# frame
	info = [
		("Inside", totalUp),
		("Outside", totalDown),
		("Status", status),
	]

	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)



	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("elapsed time: {:.2f}".format(fps.elapsed()))
print("approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer

# if we are not using a video file, stop the camera video stream
vs.stop()


# close any open windows
cv2.destroyAllWindows()
print(';;;')
print(';;;')
print(';;;')
print(';;;')
print(';;;')
print(';;;')
print(';;;')
print(';;;')