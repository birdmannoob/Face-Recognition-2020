# FaceRecog(opencv + dlib).py
#Input file paths:
#face_detection_model/deploy.prototxt
#face_detection_model/res10_300x300_ssd_iter_140000.caffemodel
#output/recognizer.pickle
#output/le.pickle
import face_recognition
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import pickle
import time
import cv2
import os
names = []  	#lists to store data
probs = []
locations = []
frame_count = 0 #counter for every frame

print("[STAT] loading face detector...")
# load the face detector
protoPath = ("face_detection_model/deploy.prototxt")
modelPath = ("face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[STAT] loading face recognizer...")
recognizer = pickle.loads(open('output/recognizer.pickle', "rb").read())
le = pickle.loads(open('output/le.pickle', "rb").read())

print("STAT] starting video stream...")
# initialize the video stream, allowing warming up of camera sensor
vs = VideoStream(src=0).start()
time.sleep(2.0)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v') #This the codec is for writing frame as video
# out = cv2.VideoWriter('Task1_video.mp4',fourcc,20.0, (640, 480))
# start the FPS throughput estimator
fps = FPS().start()
# loop over frames from the video file stream
while True:
	frame = vs.read()
	frame = cv2.resize(frame, (0, 0), fx=1 / 2, fy=1 / 2)
	(h, w) = frame.shape[:2]
	#<============================Face Detection BELOW============================>
	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
	# apply OpenCV's deep learning-based face detector
	detector.setInput(imageBlob)
	detections = detector.forward()
	boxes = []
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (probability) of the prediction
		confidence = detections[0, 0, i, 2]
		if confidence > 0.8: # filter out weak detection
			# compute the (x, y)-coordinates of the bounding box for face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			boxes.append((startY, endX, endY, startX))

	#<===========================Face Recognition BELOW===========================>
	if frame_count%4 == 0: #only perform recognition once every 4 frames
		locations = []
		names = []
		probs = []  # clear the lists from previous frame
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert image format
		#extract face 128-d embeddings using face_recognition function
		encodings = face_recognition.face_encodings(rgb, boxes)
		count = 0
		for startX, startY, endX, endY in boxes:
			# perform SVM classification to recognize the face detected
			preds = recognizer.predict_proba(encodings)[count]
			j = np.argmax(preds) #get the index of the match with highest probability
			proba = preds[j]
			if proba > 0.8: #thresholding
				name = le.classes_[j]
				prob = proba
			else:
				name = "unknown"
				prob = 0
			names.append(name)  #append data to use for next frame
			probs.append(prob)
			locations.append((startX, startY, endX, endY))
			count += 1

	# <===========================Draw out everything===========================>
	for (startY, endX, endY, startX), name, prob in zip(locations, names, probs):
		if name == "unknown":
			text = name
		else:
			text = "{}: {:.2f}%".format(name, prob * 100)

		cv2.rectangle(frame, (startX, startY), (endX, endY),
					  (0, 255, 0), 2)
		cv2.rectangle(frame, (startX, startY - 15), (endX, startY),
					  (0, 255, 0), cv2.FILLED)
		cv2.putText(frame, text, (startX + 1, startY - 4),
					cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

	#<===========================Finally, show frame===========================>
	fps.update()		# update the FPS counter
	frame_count += 1 	#update frame count
	cv2.imshow("Frame", cv2.resize(frame, (0, 0), fx= 1.5, fy= 1.5))
	#out.write(frame)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

fps.stop()	# stop the timer and display FPS information
print("[STAT] elasped time: {:.2f}".format(fps.elapsed()))
print("[STAT] approx. FPS: {:.2f}".format(fps.fps()))
# cleanup
vs.stop()
#out.release()
cv2.destroyAllWindows()



