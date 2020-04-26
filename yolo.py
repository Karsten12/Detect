import cv2
import numpy as np
import json
import time

# import custom files
import utils

# Initialize the parameters
classes = None
confThreshold = 0.7  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = inpHeight = 416  # Height/Width of network's input image
frame = None


def getOutputsNames(net):
    """ Get the names of the output layers
    
    Arguments:
        net {Net object} -- Darknet neural network
    
    Returns:
        [type] -- names of the output layers
    """
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def drawPred(classId, conf, left, top, right, bottom):
    """ Draw the predicted bounding box
    
    Arguments:
        classId {int} -- A number from 0 to 79, representing the class detected
        conf {float} -- How confident the NN that the object is actually the class detected
        left {[type]} -- The left of the bounding box
        top {[type]} -- The top of the bounding box
        right {[type]} -- The right of the bounding box
        bottom {[type]} -- The bottom of the bounding box
    """

    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = "%.2f" % conf

    # Get the label for the class name and its confidence
    if classes:
        assert classId < len(classes)
        label = "%s:%s" % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(
        frame,
        (left, top - round(1.5 * labelSize[1])),
        (left + round(1.5 * labelSize[0]), top + baseLine),
        (255, 255, 255),
        cv2.FILLED,
    )
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


def postprocess(frame, outs, show_frame=False, save_image=False):
    """ Remove the bounding boxes with low confidence using non-maxima suppression
    
    Arguments:
        frame {[type]} -- The image containing the detected object
        outs {[type]} -- The output of the yolov3 neural net for this frame
    
    Keyword Arguments:
        show_frame {bool} -- Whether to show/display frames (default: {False})
        save_image {bool} -- Whether to save images of detected objects (default: {False})
    """
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # non maximum suppression to eliminate redundant overlapping boxes with lower confidences
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        # Skip classes that aren't people
        if classIds[i] != 2:
            continue
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        if save_image:
            class_name = classes[classIds[i]]
            dimensions = (top, top + height, left, left + width)
            utils.write_image(frame, "output/yolo", class_name, dimensions)
        if show_frame:
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)


def run_yolo(net, cap, coco_classes, duration=None, show_frame=False, save_image=False):
    """ Run Yolov3 algorithm for on the cap video feed, detecting classes given by coco_classes for duration time
    
    Arguments:
        net {Net object} -- Darknet neural network
        cap {VideoCapture object} -- [description]
        coco_classes {List} -- [description]
        duration {time} -- [description]
    
    Keyword Arguments:
        show_frame {bool} -- Whether to show/display frames (default: {False})
        save_image {bool} -- Whether to save images of detected objects (default: {False})
    
    Raises:
        IOError: Video feed is not working
    """

    global frame, classes
    # Give the configuration and weight files for the model and load the network using them.
    classes = coco_classes

    if not cap.isOpened():
        raise IOError("Couldn't open video")

    if show_frame:
        cv2.namedWindow("Yolo", cv2.WINDOW_NORMAL)

    while cv2.waitKey(1) < 0:

        if duration:
            if time.time() > duration:
                break

        # Grab frame from video
        for _ in range(0, 24):
            cap.grab()
        hasFrame, frame = cap.read()

        # Crop frame -> [y_min:y_max, x_min:x_max]
        frame = frame[300:1080, 200:1920]  # Classifying people
        # frame = frame[0:500, 0:1920]  # Classifying Cars

        # Stop the program if reached end of video
        if frame is None:
            break

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False
        )

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        postprocess(frame, outs, show_frame, save_image)

        # Get the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        # t, _ = net.getPerfProfile()
        # label = "Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency())
        # cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        # print(label)

        if show_frame:
            cv2.imshow("Yolo", frame)

    # Sync up the camera back to the latest frame
    # retval = 1
    # while retval:
    #     retval = cap.grab()


if __name__ == "__main__":

    # Load details
    with open("config/config.json") as f:
        config_dict = json.load(f)

    # Load links to ip cams
    ip_cams = config_dict["ip_cams"]

    if ip_cams is None:
        utils.print_err("ip cams is none")
        exit()

    show_frames = config_dict["show_frames"]

    # Process inputs
    cap = cv2.VideoCapture(str(ip_cams[1]))

    # Load details for darknet
    modelConfiguration = "config/yolov3.cfg"
    modelWeights = "config/yolov3.weights"

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    run_yolo(net, cap, config_dict["coco_classes"], None, show_frames, False)
