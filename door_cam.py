import imutils
import cv2
import time
import numpy as np
import logging

# import custom files
import lib.utils as utils
import lib.telegram_bot as tg_bot
import lib.tflite_detect as tflite

# Check if person at sidewalk
def tflite_detection(detector_obj, frame, thresh):
    """ Checks if a person is present in the frame

    Args:
        detector_obj {Detector} -- Instance of Detector
        frame ([type]): [description]
        thresh ([type]): [description]
    """
    person_sidewalk = tflite.detect_people(detector_obj.tf_intepreter, frame, thresh)
    if person_sidewalk:
        logging.info("Person detected @ sidewalk")
        detect_person(detector_obj)
        # utils.write_frame_and_thresh(frame, thresh, True)
    else:
        logging.info("No person detected at sidewalk")
        # utils.write_frame_and_thresh(frame, thresh)


def detect_person(detector_obj):
    """ Detects motion from the door cam video feed in detector_obj, and does facial recognition 
	
	Args:
		detector_obj {Detector} -- Instance of Detector
	"""
    # (called in a new thread from motion detector)
    # Repeat following until find person with face, or timeout
    # Read each frame from door cam
    # Crop into door portion
    # Find person
    # Find face using MTCNN
    # Once face is found
    # Do sklearn SVM, check if family or not

    cap = detector_obj.ip_cam_objects["door"].start()

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
        logging.info("Person detected @ front door")

        # TODO detect face
        # Face = None
        # if not Face:
        #     continue
        # else:
        #     # Do facial recognition
        #     print("Hi")
        #     if not known face
        #     send_telegram

    cap.stop()
    return


def send_telegram(frame, thresh):
    # TODO
    # Send message via telegram
    tg_bot.send_message()
