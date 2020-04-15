# import the necessary packages
import imutils
import cv2
import json

# Load details
config_file = 'config.json'
with open(config_file) as f:
	config_dict = json.load(f)

# Load links to ip cams
ip_cams = config_dict['ip_cams']

min_area = 500

# if the video argument is None, then we are reading from webcam
if ip_cams is None:
	exit()

vs = cv2.VideoCapture(str(ip_cams[1]))


# initialize the first frame in the video stream
firstFrame = None

# loop over the frames of the video
while True:

	# grab the current frame
	status, frame = vs.read()

	# if the frame could not be grabbed, exit
	if frame is None:
		break

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < min_area:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
vs.release()
cv2.destroyAllWindows()