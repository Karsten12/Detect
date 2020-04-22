import cv2
import imutils
from datetime import datetime



def crop_and_resize_frame(frame, crop_dimensions=(175, 1080, 250, 1920)):
    """ Crop unimportant parts of frame, then resizes. Default crop detects people

    Arguments:
        frame {nd_array} -- Image frame

    Keyword Arguments:
        crop_dimensions {tuple} -- The dimensions to crop the frame (default: {(175, 1080, 250, 1920)})

    Returns:
        nd_array -- resulting image frame
    """
    y_min, y_max, x_min, x_max = crop_dimensions
    return imutils.resize(frame[y_min:y_max, x_min:x_max], width=500)

def write_image(frame, class_name=None, dimensions=None):
    """ Writes the frame as a png file
    
    Arguments:
        frame {[type]} -- The image containing the detected object
    
    Keyword Arguments:
        class_name {str} -- The predicted class (default: {None})
        dimensions {tuple} -- tuple giving (top, bottom, left and right) dimensions needed to crop the frame (default: {None})
    """

    fileName = datetime.now().strftime("%m-%d-%Y--%H-%M-%S") + ".png"
    if class_name:
        fileName = class_name + '_' + fileName
    outFile = frame
    if dimensions:
        top, bottom, left, right = dimensions
        outFile = frame[top:bottom, left:right]
    cv2.imwrite(fileName, outFile)
