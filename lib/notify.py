import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

carriers = {
    "sprint_SMS": "@messaging.sprintpcs.com",
    "sprint_MMS": "@pm.sprint.com",
}


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
