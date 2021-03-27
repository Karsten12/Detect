import imutils
import cv2
import time
import numpy as np
import threading

# import custom files
import lib.utils as utils
import lib.TFlite_detect as tflite
from lib.videostream import VideoStream
import door_cam


def motion_detector(detector_obj):
    """ Detects motion from the video feed of a single camera in detector_obj, and if motion calls YoloV3 to do object recognition 
	
	Arguments:
		detector_obj {Detector} -- Instance of Detector
	"""
    cap = detector_obj.ip_cam_objects["street"].start()

    # initialize the frame in the video stream
    avg = None

    motion_count = 0
    min_motion_frames = 20  # min num of frames w/ motion to trigger detection
    delta_thresh = 10  # min value pixel must be to trigger movement
    min_area = 50  # min area to trigger detection

    # Read in mask
    mask_image = cv2.imread("images/new_mask.png", cv2.IMREAD_GRAYSCALE)
    im_mask = utils.crop_and_resize_frame(mask_image)

    # loop over the frames of the video
    while True:
        # grab current frame
        frame = cap.read()

        # if frame not available, exit
        if frame is None:
            break

        # Assume no motion
        motion = False

        # Crop, resize and convert frame to grayscale
        cropped_frame = utils.crop_and_resize_frame(frame)
        frame_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        # Mask out the areas that are not important
        masked_frame = cv2.bitwise_and(frame_gray, frame_gray, mask=im_mask)
        # Blur
        blurred_frame = cv2.GaussianBlur(masked_frame, (21, 21), 0)

        # if the average frame is None, initialize it
        if avg is None:
            avg = blurred_frame.copy().astype("float")
            continue

        # Do weighted average between current & previous frame
        # Then compute difference between curr and average frame
        cv2.accumulateWeighted(blurred_frame, avg, 0.5)
        frameDelta = cv2.absdiff(blurred_frame, cv2.convertScaleAbs(avg))

        # Set pixels w/ values >= delta_thresh to white (255) else set to black (0)
        thresh = cv2.threshold(frameDelta, delta_thresh, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            if cv2.contourArea(c) >= min_area:
                # Motion detected
                motion = True
                break

        if motion:
            motion_count += 1
        else:
            motion_count = 0

        if motion_count >= min_motion_frames:
            # Do detection
            cap.pause()
            # Check for persion in a seperate thread, in case
            # this takes longer than 15 seconds
            async_detection(detector_obj, frame, thresh)
            # Limit the detections to max once every 20 seconds (to eliminate duplicate detections)
            # Sleep this thread for 20 seconds
            time.sleep(time.time() + 20)
            cap.resume()
            motion_count = 0

    # cleanup the camera and close any open windows
    cap.stop()


# Starts a new thread to check if person at sidewalk
def async_detection(detector_obj, frame, thresh):
    my_args = {
        "detector_obj": detector_obj,
        "frame": frame.copy(),
        "thresh": thresh.copy(),
    }
    t = threading.Thread(
        target=door_cam.tflite_detection,
        name="person-sidewalk-thread",
        kwargs=my_args,
        daemon=True,
    )
    t.start()
