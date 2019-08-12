import cv2

casPath = "C:\\Users\\m1049227\\Documents\\pplCOUNTER\\PeopleCounterPyimage_myversion2\\casscade\\"

faceCascade = cv2.CascadeClassifier(casPath)

imagePath = "C:\\Users\\Documents\\pplCOUNTER\\PeopleCounterPyimage_myversion2\\"

image = cv2.imread(imagePath)
image = image.astype('uint8')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors = 5,minSize=(30,30),flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

print("found {} Faces").format(len(faces))

for(x,y,w,h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)