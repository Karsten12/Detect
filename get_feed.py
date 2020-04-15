from threading import Thread
import cv2, datetime, time

class VideoStreamWidget(object):
    def __init__(self, src=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)

        # self.fps = int(self.capture.get(cv2.CAP))

        # self.capture.set(cv2.CAP_PROP_FPS, int(1))

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
            cv2.imshow('IP Camera Video Streaming', self.frame)

        # currentDT = datetime.datetime.now()
        # formated_time = currentDT.strftime("%Y-%m-%d %H:%M:%S")
        # print(formated_time)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    # Resizes a image and maintains aspect ratio
    def maintain_aspect_ratio_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
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

if __name__ == '__main__':

    # Open and read in rtsp URL for the cameras 
    with open('ip_cam_links.txt', 'r') as file:
        stream_link = file.readlines()
    stream_link = [x.strip() for x in stream_link]

    door_cam = stream_link[0]
    driveway_cam = stream_link[1]

    video_stream_widget = VideoStreamWidget(driveway_cam)
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass