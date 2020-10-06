import imutils
import cv2
import json
import time
import numpy as np
import sys
import threading

# import custom files
import lib.utils as utils
import lib.telegram_bot as telegram_bot
import lib.TFlite_detect as tflite
from lib.videostream import VideoStream


# Check if person at sidewalk
def tflite_detection(ip_cam_objects, frame, thresh):
    """ Checks if a person is present in the frame

    Args:
        ip_cam_objects ([type]): [description]
        frame ([type]): [description]
        thresh ([type]): [description]
    """
    person_sidewalk = tflite.detect_people(frame, thresh)
    if person_sidewalk:
        print("Person detected @ sidewalk")
        detect_person(ip_cam_objects)
        # utils.write_frame_and_thresh(frame, thresh, True)
    else:
        print("No person detected @ sidewalk")
        # utils.write_frame_and_thresh(frame, thresh)


def detect_person(ip_cam_objects):
    """ Detects people from the video feed of a single camera in ip_cam_objects, and does facial recognition 
	
	Args:
		ip_cam_objects {dict} -- Dictionary of videostream objects, representing each ip cam
	"""
    # (called in a new thread from motion detector)
    # Pass in the TFlite detector from motion_detector.py
    # Repeat following until find person with face, or timeout
    # Read each frame from door cam
    # Crop into door portion
    # Find person
    # Find face using MTCNN
    # Once face is found
    # Do sklearn SVM, check if family or not

    cap = ip_cam_objects["garage_cam"].start()

    timeout = time.time() + 20
    skip_frame = False

    while time.time() < timeout:
        # Skip every other frame (for performance)
        if skip_frame:
            skip_frame = False
            continue

        # Grab frame
        frame = cap.read()

        # Crop frame to front door area (y_min, y_max, x_min, x_max)
        cropped_frame = utils.crop_and_resize_frame(frame, (0, 700, 1100, 1920))

        # Do person detection
        # TODO return person frame
        person = tflite.detect_people(cropped_frame)
        if not person:
            continue

        # Person detected, find face using MTCNN
        print("Person detected @ front door")

        # TODO detect face
        # Face = None
        # if not Face:
        #     continue
        # else:
        #     # Do facial recognition
        #     print("Hi")
        #     if not known face
        #     send_sms_async

    cap.stop()
    return


def send_telegram(frame, thresh):
    # TODO
    # Send message via telegram
    telegram_bot.send_message()
