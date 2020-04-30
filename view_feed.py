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
            self.frame = self.maintain_aspect_ratio_resize(self.frame, width=800)
            cv2.imshow("IP Camera Video Streaming", self.frame)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord("q"):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    # Resizes a image and maintains aspect ratio
    def maintain_aspect_ratio_resize(
        self, image, width=None, height=None, inter=cv2.INTER_AREA
    ):
        # Grab the image size and initialize dimensions
        dim = None
        (h, w) = image.shape[:2]

        # Return original image if no need to resize
        if width is None and height is None:
            return image

        # We are resizing height if width is none
        if width is None:
            # Calculate the ratio of the height and construct the dimensions
            r = height / float(h)
            dim = (int(w * r), height)
        # We are resizing width if height is none
        else:
            # Calculate the ratio of the 0idth and construct the dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # Return the resized image
        return cv2.resize(image, dim, interpolation=inter)


if __name__ == "__main__":

    # Open and read in rtsp URL for the cameras
    with open("config/config.json") as f:
        config_dict = json.load(f)
    ip_cams = config_dict["ip_cams"]

    door_cam = ip_cams[0]
    driveway_cam = ip_cams[1]

    video_stream_widget = VideoStreamWidget(driveway_cam)
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass
