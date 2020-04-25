import sys
import cv2
import imutils
from datetime import datetime


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

    # TODO 
    return None


def crop_and_resize_frame(frame, crop_dimensions=(200, 1080, 250, 1920)):
    """ Crop unimportant parts of frame, then resizes. Default crop detects people

    Arguments:
        frame {nd_array} -- Image frame

    Keyword Arguments:
        crop_dimensions {tuple} -- The dimensions to crop the frame (default: {(175, 1080, 250, 1920)})

    Returns:
        nd_array -- resulting image frame
    """
    y_min, y_max, x_min, x_max = crop_dimensions
    # Crop frame -> [y_min:y_max, x_min:x_max]
    return imutils.resize(frame[y_min:y_max, x_min:x_max], width=500)


def write_image(frame, directory=None, class_name=None, dimensions=None):
    """ Writes the frame as a png file
    
    Arguments:
        frame {[type]} -- The image containing the detected object
    
    Keyword Arguments:
        directory {str} -- The directory to store class (default: current directory)
        class_name {str} -- The predicted class (default: {None})
        dimensions {tuple} -- tuple giving (top, bottom, left and right) dimensions needed to crop the frame (default: {None})
    """

    fileName = datetime.now().strftime("%m-%d-%Y--%H-%M-%S") + ".png"
    if class_name:
        fileName = class_name + "_" + fileName
    if directory:
        fileName = directory + "/" + fileName
    outFile = frame
    if dimensions:
        top, bottom, left, right = dimensions
        outFile = frame[top:bottom, left:right]
    cv2.imwrite(fileName, outFile)
