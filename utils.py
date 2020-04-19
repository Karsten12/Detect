import cv2
from datetime import datetime


def write_image(frame, class_name=None, dimensions=None):
    """ Writes the frame as a png file
    
    Arguments:
        frame {[type]} -- The image containing the detected object
        class_name {str} -- The predicted class
        dimensions {tuple} -- 4d tuple representing the top, bottom, left and right dimensions needed to crop the frame
    """
    fileName = datetime.now().strftime("%m-%d-%Y--%H-%M-%S") + ".png"
    if class_name:
        fileName = str(class_name) + fileName
    outFile = frame
    if dimensions:
        top, bottom, left, right = dimensions
        outFile = frame[top:bottom, left:right]
    cv2.imwrite(fileName, outFile)
