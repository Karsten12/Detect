import sys
import cv2
import imutils
from datetime import datetime

y_crop = 200  # Crop in the image 200 from the top
x_crop = 250  # Crop in the image 250 from the left


def print_err(out):
    """ Print out an a string to stderr

    Arguments:
        out {str} -- The string to print
    """
    print(out, file=sys.stderr)


def get_padding_detection(frame, thresh):
    """ Return a locally cropped area (padded w/ .. ) of the motion detected in thresh

    Arguments:
        frame {[type]} -- [description]
        thresh {[type]} -- [description]
    """
    frame_im = cv2.imread(frame)
    thresh_im = cv2.imread(thresh, cv2.IMREAD_GRAYSCALE)

    upscaled_width = 1920 - x_crop
    new_thresh_im = imutils.resize(thresh_im, width=upscaled_width)

    # Calculate the coordinates of the detection
    cnts = cv2.findContours(
        new_thresh_im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Get the bounding box from the coordinates
    x, y, w, h = cv2.boundingRect(cnts[0])
    # Move bounding box by to compensate for crop
    x += x_crop
    y += y_crop
    # Pad bounding box
    x_pad, y_pad = 100, 125
    y_min, y_max, x_min, x_max = (y - y_pad, y + h + y_pad, x - x_pad, x + w + x_pad)

    # Crop out the rectangle and write it
    cropped_object = frame_im[y_min:y_max, x_min:x_max]
    write_image(cropped_object)

    # Display rectangle (w/ padding)
    # im = cv2.rectangle(frame_im, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
    # cv2.imshow("temp", im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # TODO
    return None


def crop_and_resize_frame(frame, crop_dimensions=(y_crop, 1080, x_crop, 1920)):
    """ Crop unimportant parts of frame, then resizes. Default crop detects people

    Arguments:
        frame {nd_array} -- Image frame

    Keyword Arguments:
        crop_dimensions {tuple} -- The dimensions to crop the frame (default: {(200, 1080, 250, 1920)})

    Returns:
        nd_array -- resulting image frame
    """
    y_min, y_max, x_min, x_max = crop_dimensions
    # Crop frame -> [y_min:y_max, x_min:x_max]
    return imutils.resize(frame[y_min:y_max, x_min:x_max], width=500)


def write_frame_and_thresh(frame, thresh):
    curr_time = datetime.now().strftime("%m-%d-%Y--%H-%M-%S")
    write_image(frame, directory="output/motion", time=curr_time)
    write_image(thresh, directory="output/motion", class_name="thresh", time=curr_time)


def write_image(frame, directory=None, class_name=None, dimensions=None, time=None):
    """ Writes the frame as a png file
    
    Arguments:
        frame {[type]} -- The image containing the detected object
    
    Keyword Arguments:
        directory {str} -- The directory to store class (default: current directory)
        class_name {str} -- The predicted class (default: {None})
        dimensions {tuple} -- tuple giving (top, bottom, left and right) dimensions needed to crop the frame (default: {None})
    """

    if time:
        fileName = time
    else:
        fileName = datetime.now().strftime("%m-%d-%Y--%H-%M-%S")
    if class_name:
        fileName += "_" + class_name
    if directory:
        fileName = directory + "/" + fileName
    fileName += ".png"
    outFile = frame
    if dimensions:
        top, bottom, left, right = dimensions
        outFile = frame[top:bottom, left:right]
    cv2.imwrite(fileName, outFile)
