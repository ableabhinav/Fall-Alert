#Things needed in GUI:
# 1) Receiver email
# 2) message
# 3) Video input path
# 4) Execute button
# USAGE python yolo_video.py --input test.mp4 --yolo yolo-coco
# Better Classifier for only Person
# How to modify code to only detect person
import numpy as np
import argparse
import imutils
import time
import cv2
import os

import smtplib 
  
# creates SMTP session 
s = smtplib.SMTP('smtp.gmail.com', 587) 
  
# start TLS for security 
s.starttls() 
  
# Authentication 
s.login("ritik.posh@gmail.com", "$HPau620tx") 
  
# message to be sent 
message = "Fall has been Detected. Please take any appropriate action"
msg=0  

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=False,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
# labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")  #Changes here

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()   #Changes here
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1
cnt = 0 
fno = 0
# ------------------FRAME PART-----------------------------------------
counter = 0
start2 = time.time()
while True:
	start1 = time.time()
	(grabbed, frame) = vs.read()
	fno+=1
	if fno%60!=0:
		continue
	print("Frame No:", fno)
	if not grabbed:
		break
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))     #Top-Left Co-ordinates
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)			

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])
	#idxs contains indices that can be used from boxes, classIDs lists
	# ensure at least one detection exists
	if len(idxs) > 0:
		idArray = []
		for j in idxs.flatten():
			if classIDs[j]==0:
				idArray.append(j)
	# 	#Intersection of bounding box code
		print("Dimensions:")
	# boxes, idxs, idArray
		print(f"For Boxes:: Length: {len(boxes)} Values: {boxes}")
		print(f"For idxs:: Length: {idxs.size} Values: {idxs.flatten()}")
		print(f"For idArray:: Length: {len(idArray)} Values: {idArray}")
		for i in idArray:
		# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			if(w>h):
			# draw a bounding box rectangle and label on the frame
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				text = "FALL ALERT!! {}: {:.4f}".format(LABELS[classIDs[i]],
					confidences[i])
				cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				cv2.imwrite("output-video\\Alert{}.jpg".format(fno), frame)
				# sending the mail
				msg+=1
				if msg==2: 
					s.sendmail("enter your mail id here", message)
				print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
				file=open('fallInfo1.txt','a')
				file.write(f"Fall_Detected{time.time()}-Fall-Alert{fno}.jpg\n")
			else:
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]],
					confidences[i])
				cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				cv2.imwrite("output-video\\Frame{}.jpg".format(fno), frame)
			print(f"Box[{i}]: {x} {y} {w} {h} Labels[{i}]: {LABELS[classIDs[i]]} classIDs[{i}]: {classIDs[i]}  confidences[{i}]: {confidences[i]}")			
	end1 = time.time()
	# cv2.imwrite("output-image/frame{}.jpg".format(fno), frame)
	print("Complete time for algorithm", (end1-start1))
	cnt+=1
print("[INFO] cleaning up...")
end2 = time.time()
print(end2-start2)
vs.release()