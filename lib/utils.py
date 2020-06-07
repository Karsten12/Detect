import sys
import cv2
import imutils
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

# For a 1920 x 1080 (1080p) feed
frame_max_x, frame_max_y = 1920, 1080
frame_min = 0

# 0 <= Y <= 1080
y_crop_min = 200  # start crop 200 from the top
y_crop_max = 650  # end crop 650 from the top

# 0 <= X <= 1920
x_crop_min = 450  # start crop 450 from the left (NEW MASK), 250 is for OLD MASK

carriers = {
    "sprint_SMS": "@messaging.sprintpcs.com",
    "sprint_MMS": "@pm.sprint.com",
}


def print_err(out):
    """ Print out an a string to stderr

    Arguments:
        out {str} -- The string to print
    """
    print(out, file=sys.stderr)


def get_padding_detection(frame, thresh):
    """ Return a locally cropped area (padded w/ .. ) of frame using motion detected in the binary image thresh

    Arguments:
        frame {nd_array} -- Image frame
        thresh {nd_array} -- The detection frame
    """

    # Scale up (back) the thresh image to its original size post crop
    upscaled_width = 1920 - x_crop_min
    new_thresh_im = imutils.resize(thresh, width=upscaled_width)

    # Calculate the coordinates of the detection
    cnts = cv2.findContours(
        new_thresh_im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Get the bounding box from the coordinates
    x, y, w, h = cv2.boundingRect(cnts[0])
    # Move bounding box to compensate for crop
    x += x_crop_min
    y += y_crop_min
    # Pad bounding box
    x_pad, y_pad = 100, 125
    # Ensure that padding doesn't extend beyond frame
    y_min, y_max = max(y - y_pad, frame_min), min(y + h + y_pad, frame_max_y)
    x_min, x_max = max(x - x_pad, frame_min), min(x + w + x_pad, frame_max_x)

    # Crop out the rectangle
    cropped_object = frame[y_min:y_max, x_min:x_max]

    # Add bounding box (w/ padding) to image
    # im = cv2.rectangle(
    #     cropped_object, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2
    # )
    # cv2.imshow("temp", im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return cropped_object

    # Return image w/ bounding box
    # return im


def crop_and_resize_frame(
    frame, crop_dimensions=(y_crop_min, y_crop_max, x_crop_min, frame_max_x)
):
    """ Crop unimportant parts of frame, then resizes. Default crop detects people

    Arguments:
        frame {nd_array} -- Image frame

    Keyword Arguments:
        crop_dimensions {tuple} -- The dimensions to crop the frame (default: {(y_crop_min, y_crop_max, x_crop_min, frame_max_x)})

    Returns:
        nd_array -- resulting image frame
    """
    y_min, y_max, x_min, x_max = crop_dimensions
    # Crop frame -> [y_min:y_max, x_min:x_max]
    return imutils.resize(frame[y_min:y_max, x_min:x_max], width=500)


def write_frame_and_thresh(frame, thresh, people=False):
    curr_time = datetime.now().strftime("%m-%d-%Y--%H-%M-%S")
    if people:
        write_image(
            frame, directory="output/motion", class_name="person", time=curr_time
        )
        write_image(
            thresh,
            directory="output/motion",
            class_name="person_thresh",
            time=curr_time,
        )
    else:
        write_image(frame, directory="output/motion", time=curr_time)
        write_image(
            thresh, directory="output/motion", class_name="thresh", time=curr_time
        )


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


def login_server(auth):
    # Establish a secure session with gmail's outgoing SMTP server using your gmail account
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(auth[0], auth[1])

    return server


def get_recipients_from_numbers(phone_numbers):
    formatted_numbers = []
    for number in phone_numbers:
        tmp = str(number) + carriers["sprint_SMS"]
        formatted_numbers.append(tmp)

    return formatted_numbers


def send_message(auth, recipients, frame=None, thresh=None):

    server = login_server(auth)

    to_list = get_recipients_from_numbers(recipients)

    msg = MIMEMultipart()

    text = "Motion detected at {}".format(datetime.now().strftime("%m-%d-%Y--%H-%M-%S"))

    # Include text
    msg.attach(MIMEText(text))

    # If image, include it
    if frame:
        img = get_padding_detection(frame, thresh)
        img_str = cv2.imencode(".png", img)[1].tostring()
        msg.attach(MIMEImage(img_str))

    msg_to_send = msg.as_string()

    # Send SMS/MMS
    server.sendmail(auth[0], to_list, msg_to_send)
    server.quit()
