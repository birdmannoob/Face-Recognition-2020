# Input file(s) for this script (encode_faces.py) :
# dataset NOTE: 'dataset' is the directory containing all training images
#Output file(s) for this script:
#output/encodings encodings.pickle

from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import dlib
# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images('dataset'))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	#convert image from BGR (OpenCV format)
	# to RGB (dlib format)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,model='hog')
	#another mode is 'cnn' mode for cuda accelerated GPU
	#print(boxes)
	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)
		print(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open('output/encodings.pickle', "wb")
f.write(pickle.dumps(data))
f.close()




