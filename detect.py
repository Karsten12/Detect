import imutils
import cv2
import numpy as np
import threading
import logging

# import custom files
import lib.utils as utils
import lib.tflite_detect as tflite
from lib.videostream import VideoStream
import lib.telegram_bot as tg_bot
import motion_detector as md


class Detector:
    def __init__(
        self, ip_cam_objects, tf_intepreter, telegram_people, telegram_token
    ):
        """[summary]

        Args:
            ip_cam_objects (Dict): Dictionary of videostream objects, representing each ip cam
            tf_intepreter (tflite_runtime.interpreter): Instance of tflite_runtime.interpreter
            telegram_people (Dict): Contains name:id pairs of all authorized telegram users
            telegram_token (String): The token needed to use the bot
        """
        self.ip_cam_objects = ip_cam_objects
        self.tf_intepreter = tf_intepreter
        self.telegram_people = telegram_people
        self.telegram_ids = set(telegram_people.values())
        self.telegram_token = telegram_token


if __name__ == "__main__":
    # --- Load options from config ---
    config_dict = utils.load_config()

    ip_cams = config_dict["ip_cams"]  # load up list of ip_cams
    telegram_token = config_dict["telegram_token"]  # load token for telegram bot
    people = config_dict["people"]  # Load up dict of people for telegram bot

    # Create the videostream objects for each ip cam
    ip_cam_objects = {
        ip_cam: VideoStream(ip_cams[ip_cam], ip_cam) for ip_cam in ip_cams
    }

    # Pre-load tf model
    tf_intepreter = tflite.load_tflite_model()

    # Create detector object
    detector_obj = Detector(ip_cam_objects, tf_intepreter, people, telegram_token)

    # Create telegram thread
    threading.Thread(
        target=tg_bot.poll,
        name="Telegram-poll-thread",
        args=(detector_obj,),
        daemon=True,
    ).start()

    logging.info("Starting motion detection...")
    md.motion_detector(detector_obj)
