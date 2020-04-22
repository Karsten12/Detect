import imutils
import cv2
import json
import time
from datetime import datetime
import numpy as np
import sys

# import custom files
import yolo
import utils



crop_dimensions=(175, 1080, 250, 1920)
def crop_and_resize_frame(frame):
    # Default crop detects people
    y_min, y_max, x_min, x_max = crop_dimensions
    return imutils.resize(frame[y_min:y_max, x_min:x_max], width=500)

def crop_and_or_resize_frame(frame, crop=False, resize=False):
    # Default crop detects people
    if not crop and not resize:
        exit()
    if crop:
        y_min, y_max, x_min, x_max = crop_dimensions
        temp = frame[y_min:y_max, x_min:x_max]
    if resize:
        if crop:
            return imutils.resize(temp, width=500)
        else:
            return imutils.resize(frame, width=500)

def motion_detector(ip_cam, show_frames=False):
    """ Detects motion from the video feed of ip_cam, and if motion calls YoloV3 to do object recognition 
	
	Arguments:
		ip_cam {str} -- The rtsp url to the live camera feed
	
	Keyword Arguments:
		show_frames {bool} -- Whether or not to display the feed and the output (default: {False})
	"""

    cap = cv2.VideoCapture(ip_cam)

    # initialize the frame in the video stream
    avg = None

    motionCounter = 0
    min_motion_frames = 20 # min number of frames with motion needed to trigger detection 
    delta_thresh = 10 # min value a pixel must be to trigger movement
    min_area = 50 # min area to trigger detection
    blur_kernel = (21, 21)
    write_timeout = 0

    kernel = np.ones((3, 3), np.uint8)

    # Read in mask 
    mask_image = cv2.imread('images/mask.png', cv2.IMREAD_GRAYSCALE)
    im_mask = crop_and_resize_frame(mask_image)

    # loop over the frames of the video
    while True:
        # grab current frame
        status, frame = cap.read()

        # if frame not available, exit
        if frame is None:
            break

        # Assume no motion
        motion = False

        # Resize and convert frame to grayscale
        cropped_frame = crop_and_resize_frame(frame)
        frame_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
 
        # Mask out the areas that are not important 
        frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask = im_mask)
        # Blur 
        frame_gray = cv2.GaussianBlur(frame_gray, blur_kernel, 0)

        # if the average frame is None, initialize it
        if avg is None:
            avg = frame_gray.copy().astype("float")
            continue

        # Do weighted average between current & previous frame
        # Then compute difference between curr and average frame
        cv2.accumulateWeighted(frame_gray, avg, 0.5)
        frameDelta = cv2.absdiff(frame_gray, cv2.convertScaleAbs(avg))

        # Set pixels w/ values >= delta_thresh to white (255) else set to black (0)
        thresh = cv2.threshold(frameDelta, delta_thresh, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        # thresh = cv2.dilate(thresh, kernel, iterations=2)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < min_area:
                continue

            motion = True

            # compute the bounding box for the contour, draw it on the frame,
            if show_frames:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if motion:
            motionCounter += 1
        else:
            motionCounter = 0

        if motionCounter >= min_motion_frames:
            # Do YOLO detection for 15 seconds
            # print("Running Yolov3 detection")
            curr = time.time()
            if curr > write_timeout:
                print("Running Yolov3 detection")
                # Ensure only 1 image gets written every 15 seconds
                # utils.write_image(frame)
                # utils.write_image(thresh, class_name = 'thresh')
                write_timeout = curr + 20 * 1
                print("Finished Yolov3")
            # yolo_timeout = time.time() + 15 * 1
            # if use_yolov3:
            #     yolo.run_yolo(
            #         net, cap, classes, yolo_timeout, show_frame=False, store_image=True
            #     )
            # print("Finished Yolov3")
            motionCounter = 0

        # show the frame and record if the user presses a key
        if show_frames:
            feed = np.concatenate((frame_gray, thresh), axis=0)
            # Show -> resized_frame, cropped_frame, masked_frame, thresh, delta
            cv2.imshow("Security Feed", feed)

        # if the `q` key is pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # cleanup the camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # Redirect the console and error to files for later viewing
    # sys.stdout = open('out.txt', 'w')
    # sys.stderr = open('error.txt', 'w')

    # Load details
    with open("config.json") as f:
        config_dict = json.load(f)

    # Load the classes
    classes = config_dict["coco_classes"]
    # Load links to ip cams
    ip_cams = config_dict["ip_cams"]

    if ip_cams is None:
        exit()

    show_frames = config_dict["show_frames"]
    use_yolov3 = config_dict["yolov3"]

    if use_yolov3:
        modelConfiguration = "yolov3.cfg"
        modelWeights = "yolov3.weights"

        net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print("Starting motion detection...")

    # motion_detector(ip_cams[1], show_frames)
    motion_detector('vid_out_trim.mp4', show_frames)


    # #!/usr/bin/env python3 to top of file
    # chmod +x motion_detector.py
    # Run using -> nohup timeout 10 motion_detector &
