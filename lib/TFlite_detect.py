"""Utility function for object detection using TFlite"""

import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from PIL import Image

# import custom files
import lib.utils as utils

THRESHOLD = 0.0
IMAGE_WIDTH, IMAGE_HEIGHT = 300, 300
bbox_array = [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH]
tf_interpreter = None


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details["index"]))
    return tensor


def detect_objects(interpreter, image):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        # Only check for people
        if classes[i] != 0.0:
            continue
        if scores[i] >= THRESHOLD:
            result = {
                "bounding_box": boxes[i],
                "class_id": classes[i],
                "score": scores[i],
            }
            results.append(result)
    return results


def load_tflite_model():
    # Load the model
    model = "lib/SSD-mobileNET-v2.tflite"

    global tf_interpreter

    # Load up tflite
    tf_interpreter = Interpreter(model)


def detect_people(image, thresh):
    tf_interpreter.allocate_tensors()

    cropped_image = utils.get_padding_detection(image, thresh)

    # Resize and convert image to PIL format for input into model
    resized_img = cv2.resize(cropped_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(img_rgb)

    # Do detection and time it
    # start_time = time.monotonic()
    results = detect_objects(tf_interpreter, pil_im)
    # elapsed_ms = time.monotonic() - start_time
    # print(results)
    if len(results) > 0:
        # Person detected
        return True

    # No person detected
    return False

    if len(results) < 1:
        exit()


def detect(image):

    load_tflite_model()

    tf_interpreter.allocate_tensors()

    # Calculate the input height/width for the model (already known)
    # _, input_height, input_width, _ = interpreter.get_input_details()[0]["shape"]

    # Resize image for input into model
    resized_img = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Do detection and time it
    start_time = time.monotonic()
    results = detect_objects(tf_interpreter, resized_img, THRESHOLD)
    elapsed_ms = time.monotonic() - start_time

    if len(results) < 1:
        exit()

    print(results)
    print(elapsed_ms)

    # Calculate the actual bounding box
    resulting_bounding_box = results[0]["bounding_box"]  # ymin, xmin, ymax, xmax
    true_bbox = np.multiply(bbox_array, resulting_bounding_box).astype(int)
    start_point = (true_bbox[1], true_bbox[0])
    end_point = (true_bbox[3], true_bbox[2])

    # Draw bounding box on image
    new_cv_image = cv2.rectangle(
        resized_img, start_point, end_point, (255, 255, 255), 2
    )

    # Show image
    cv2.imshow("IDK", new_cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
