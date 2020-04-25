import imutils
import cv2
import json
import time
import numpy as np
import sys

# import custom files
import yolo
import utils


def motion_detector(ip_cam, show_frames=False):
    """ Detects motion from the video feed of ip_cam, and if motion calls YoloV3 to do object recognition 
	
	Arguments:
		ip_cam {str} -- The rtsp url to the live camera feed
	
	Keyword Arguments:
		show_frames {bool} -- Whether or not to display the feed and the output (default: {False})
	"""

    cap = cv2.VideoCapture(ip_cam)

    # initialize the frame in the video stream
    avg = None

    motion_count = 0
    min_motion_frames = 20  # min num of frames w/ motion to trigger detection
    delta_thresh = 10  # min value pixel must be to trigger movement
    min_area = 50  # min area to trigger detection
    blur_kernel = (21, 21)
    write_timeout = 0

    np.ones((3, 3), np.uint8)

    # Read in mask
    mask_image = cv2.imread("images/mask.png", cv2.IMREAD_GRAYSCALE)
    im_mask = utils.crop_and_resize_frame(mask_image)

    # loop over the frames of the video
    skip_frame = False
    while True:
        # Skip every other frame (for performance) until motion detected
        if skip_frame and not motion_count:
            skip_frame = False
            cap.grab()
            continue
        skip_frame = True

        # grab current frame
        status, frame = cap.read()

        # if frame not available, exit
        if frame is None:
            break

        # Assume no motion
        motion = False

        # Crop and resize and convert frame to grayscale
        cropped_frame = utils.crop_and_resize_frame(frame)
        frame_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        # Mask out the areas that are not important
        masked_frame = cv2.bitwise_and(frame_gray, frame_gray, mask=im_mask)
        # Blur
        blurred_frame = cv2.GaussianBlur(masked_frame, blur_kernel, 0)

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
        # thresh = cv2.dilate(thresh, kernel, iterations=2)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                # if the contour is too small, ignore it
                continue

            motion = True

        if motion:
            motion_count += 1
        else:
            motion_count = 0

        if motion_count >= min_motion_frames:
            # Do YOLO detection for 15 seconds
            # print("Running Yolov3 detection")
            curr = time.time()
            if curr > write_timeout:
                # Ensure only 1 image gets written every 15 seconds
                print("Writing image")
                utils.write_image(frame, directory="output/motion")
                utils.write_image(
                    thresh, directory="output/motion", class_name="thresh"
                )
                write_timeout = curr + 20 * 1
            # yolo_timeout = time.time() + 15 * 1
            # if use_yolov3:
            #     yolo.run_yolo(
            #         net, cap, classes, yolo_timeout, show_frame=False, store_image=True
            #     )
            # print("Finished Yolov3")
            motion_count = 0

        # show the frames
        if show_frames:
            # Show -> resized_frame, masked_frame, thresh,
            orig_frame_gray = cv2.cvtColor(
                imutils.resize(frame, width=500), cv2.COLOR_BGR2GRAY
            )
            feed = np.concatenate((orig_frame_gray, masked_frame, thresh), axis=0)
            # Show -> masked_frame, blurred_frame, thresh,
            # feed = np.concatenate((masked_frame, blurred_frame, thresh), axis=0)
            cv2.imshow("Security Feed", feed)

            # if the `q` key is pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    # cleanup the camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # Load details
    with open("config.json") as f:
        config_dict = json.load(f)

    # Load the classes
    classes = config_dict["coco_classes"]
    # Load links to ip cams
    ip_cams = config_dict["ip_cams"]
    # Decide whether to output logs to console or file
    logs_to_file = config_dict["logs_to_file"]

    if ip_cams is None:
        utils.print_err("ip cams is none")
        exit()

    if logs_to_file:
        # Redirect the console and error to files for later viewing
        sys.stdout = open("output/logs/out.txt", "w")
        sys.stderr = open("output/logs/error.txt", "w")

    show_frames = config_dict["show_frames"]
    use_yolov3 = config_dict["yolov3"]

    if use_yolov3:
        modelConfiguration = "yolov3.cfg"
        modelWeights = "yolov3.weights"

        net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print("Starting motion detection...")

    motion_detector(ip_cams[1], show_frames)
    # motion_detector('vid_out_trim.mp4', show_frames)
