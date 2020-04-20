import cv2
from datetime import datetime


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
