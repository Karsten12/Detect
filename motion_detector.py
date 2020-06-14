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


def motion_detector(ip_cam):
    """ Detects motion from the video feed of ip_cam, and if motion calls YoloV3 to do object recognition 
	
	Arguments:
		ip_cam {str} -- The rtsp url to the live camera feed
	"""

    cap = VideoStream(ip_cam).start()

    # initialize the frame in the video stream
    avg = None

    motion_count = 0
    min_motion_frames = 20  # min num of frames w/ motion to trigger detection
    delta_thresh = 10  # min value pixel must be to trigger movement
    min_area = 50  # min area to trigger detection
    write_timeout = 0

    # Read in mask
    mask_image = cv2.imread("images/mask_night.png", cv2.IMREAD_GRAYSCALE)
    im_mask = utils.crop_and_resize_frame(mask_image)

    skip_frame = False

    # loop over the frames of the video
    while True:
        # Skip every other frame (for performance) until motion detected
        if skip_frame and not motion_count:
            skip_frame = False
            continue
        skip_frame = True

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
            if curr > write_timeout:
                # Ensure only 1 image gets written every 20 seconds
                # cap.pause()
                print("Doing Detection")
                tflite_detection(frame, thresh)
                # cap.resume()
                write_timeout = curr + 20
            motion_count = 0

    # cleanup the camera and close any open windows
    cap.stop()


def tflite_detection(frame, thresh):
    person_sidewalk = tflite.detect_people(frame, thresh)
    if person_sidewalk:
        print("Person detected @ sidewalk")
        door_cam.detect_person(ip_cams[0])
        # utils.write_frame_and_thresh(frame, thresh, True)
    else:
        print("No person detected @ sidewalk")
        # utils.write_frame_and_thresh(frame, thresh)


def send_sms_async(frame, thresh):
    # Send SMS in another thread
    sms_args = {
        "auth": sms_auth,
        "recipients": sms_reciepients,
        "frame": frame.copy(),
        "thresh": thresh.copy(),
    }
    utils.send_message(sms_auth, sms_reciepients)
    t = threading.Thread(target=utils.send_message, kwargs=sms_args)
    t.start()


if __name__ == "__main__":

    # --- Load options from config ---
    with open("lib/config.json") as f:
        config_dict = json.load(f)
    global ip_cams
    ip_cams = config_dict["ip_cams"]  # links to ip cams

    # logs_to_file = config_dict["logs_to_file"]  # Output logs to file? (else console)
    # send_sms = config_dict["send_sms"]  # Use sms notification?

    if ip_cams is None:
        utils.print_err("ip cams is none")
        exit()

    # if logs_to_file:
    #     # Redirect the console and error to files for later viewing
    #     sys.stdout = open("output/logs/out.txt", "w")
    #     sys.stderr = open("output/logs/error.txt", "w")

    # if send_sms:
    #     sms_auth = config_dict["sms_auth"]
    #     sms_reciepients = config_dict["sms_reciepients"]

    # Pre-load tf model
    tflite.load_tflite_model()

    print("Starting motion detection...")

    motion_detector(ip_cams[1])
