import imutils
import cv2
import json
import time
import numpy as np
import sys
import threading

# import custom files
import lib.utils as utils
import lib.TFlite_detect as tflite
from lib.videostream import VideoStream
import door_cam


def motion_detector(ip_cam_objects):
    """ Detects motion from the video feed of a single camera in ip_cam_objects, and if motion calls YoloV3 to do object recognition 
	
	Arguments:
		ip_cam_objects {dict} -- Dictionary of videostream objects, representing each ip cam
	"""
    cap = ip_cam_objects["driveway_cam"].start()

    # initialize the frame in the video stream
    avg = None

    motion_count = 0
    min_motion_frames = 20  # min num of frames w/ motion to trigger detection
    delta_thresh = 10  # min value pixel must be to trigger movement
    min_area = 50  # min area to trigger detection
    timeout = 0

    # Read in mask
    mask_image = cv2.imread("images/mask_night.png", cv2.IMREAD_GRAYSCALE)
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
            if cv2.contourArea(c) < min_area:
                # if the contour is too small, ignore it
                continue
            else:
                # Motion detected
                motion = True
                break

        if motion:
            motion_count += 1
        else:
            motion_count = 0

        if motion_count >= min_motion_frames:
            # Do detection
            curr = time.time()
            if curr > timeout:
                # Limit the detections to max once every 20 seconds (to eliminate duplicate detections)
                cap.pause()
                # Do this in a seperate thread, in case
                # this takes longer than 20 seconds,
                tflite_detection(frame, thresh)
                # Sleep this thread for 20 seconds
                cap.resume()
                timeout = curr + 20
            motion_count = 0

    # cleanup the camera and close any open windows
    cap.stop()


# Check if person at sidewalk
def tflite_detection(frame, thresh):
    person_sidewalk = tflite.detect_people(frame, thresh)
    if person_sidewalk:
        print("Person detected @ sidewalk")
        door_cam.detect_person(ip_cam_objects)
        # utils.write_frame_and_thresh(frame, thresh, True)
    else:
        print("No person detected @ sidewalk")
        # utils.write_frame_and_thresh(frame, thresh)


if __name__ == "__main__":

    # --- Load options from config ---
    config_dict = utils.load_config()
    ip_cams = config_dict["ip_cams"]  # links to ip cams

    if ip_cams is None:
        utils.print_err("ip cams is none")
        exit()

    # Create the videostream objects for each ip cam
    ip_cam_objects = {}
    for ip_cam in ip_cams:
        ip_cam_objects[ip_cam] = VideoStream(ip_cams[ip_cam], ip_cam)

    # Pre-load tf model
    tflite.load_tflite_model()

    print("Starting motion detection...")
    motion_detector(ip_cam_objects)
