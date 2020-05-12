# USAGE
# python recognize_video.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle

# import the necessary packages
import face_recognition
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import pickle
import time
import cv2
import os
names = []
probs = []
locations = []
frame_count = 0
# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = ("face_detection_model/deploy.prototxt")
modelPath = ("face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open('output/recognizer.pickle', "rb").read())
le = pickle.loads(open('output/le.pickle', "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Task1_video.mp4',fourcc,20.0, (640, 480))

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:
	if frame_count%2 == 0:
		locations = []
		names = []
		probs = []  # clear the lists from previous frame
	# grab the frame from the threaded video stream
	frame = vs.read()
	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	#frame = imutils.resize(frame, width=600)
	frame = cv2.resize(frame, (0, 0), fx=1 / 2, fy=1 / 2)
	(h, w) = frame.shape[:2]
	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()


	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > 0.8:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			boxes = []
			boxes.append((startY, endX, endY, startX))
			loc = (startY, endX, endY, startX)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
						  (0, 255, 0), 2)
			cv2.rectangle(frame, (startX, startY - 15), (endX, startY),
						  (0, 255, 0), cv2.FILLED)

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue
			if frame_count%2 == 0:

				rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				encodings = face_recognition.face_encodings(rgb, boxes)
				# perform classification to recognize the face
				preds = recognizer.predict_proba(encodings)[0]
				j = np.argmax(preds)
				proba = preds[j]

				# draw the bounding box of the face along with the
				# associated probability
				if proba > 0.8:
					name = le.classes_[j]
					prob = proba

				else:
					name = "unknown"
					prob = 0

				names.append(name)
				probs.append(prob)
				locations.append(loc)

	for (startY, endX, endY, startX), name, prob in zip(locations, names, probs):
		text = "{}: {:.2f}%".format(name, prob * 100)
		#y = startY - 8 if startY - 8 > 8 else startY + 8
		cv2.putText(frame, text, (startX + 1, startY - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)



	# update the FPS counter
	fps.update()
	frame_count += 1
	# show the output frame
	cv2.imshow("Frame", frame)#cv2.resize(frame, (0, 0), fx= 1.5, fy= 1.5)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()