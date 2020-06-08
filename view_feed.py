import imutils
import argparse
from threading import Thread
import cv2
import json


class VideoStreamWidget(object):
    def __init__(self, src=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self):
        # Display frames in main program
        if self.status:
            self.frame_resized = imutils.resize(self.frame, width=600)
            cv2.imshow("IP Camera Video Streaming", self.frame_resized)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord("q"):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cam", help="Which Camera to use", type=int, default=1, required=False
    )
    args = parser.parse_args()

    # Open and read in rtsp URL for the cameras
    with open("lib/config.json") as f:
        config_dict = json.load(f)
    ip_cams = config_dict["ip_cams"]

    door_cam = ip_cams[0]
    driveway_cam = ip_cams[1]

    camera = None
    if args.cam == 1:
        camera = driveway_cam
    elif args.cam == 0:
        camera = door_cam

    if not camera:
        print("No camera specified")

    video_stream_widget = VideoStreamWidget(camera)
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass
