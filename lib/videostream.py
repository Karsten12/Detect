# import the necessary packages
from threading import Thread
import cv2


class VideoStream:
    def __init__(self, src, name):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        _, self.frame = self.stream.read()

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.paused = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, paused the thread
            if self.paused:
                return

            # otherwise, read the next frame from the stream
            _, self.frame = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def read_single_frame(self):
        # Read a single frame w/o having to start the videostream
        if not self.paused:
            return self.frame
        return self.stream.read()[1]

    def pause(self):
        # indicate that the thread should be paused
        self.paused = True

    def resume(self):
        # indicate that the thread should be paused
        self.paused = False
        self.start()

    def stop(self):
        self.pause()
        self.stream.release()
        cv2.destroyAllWindows()
