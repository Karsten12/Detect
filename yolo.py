import cv2 as cv
import sys
import numpy as np
import os.path
import json
from datetime import datetime

# Initialize the parameters
confThreshold = 0.8  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = inpHeight = 320       # Height/Width of network's input image

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classIds[i], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
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
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        # Skip classes that aren't people
        if classIds[i] != 0:
            continue
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        
        class_id = classes[classIds[i]]
        dimensions = (top, top + height, left, left+width)
        write_image(frame, class_id, dimensions)


def write_image(frame, class_id, dimensions):
    fileName = str(class_id) + '_' + str(datetime.now()) + '.png'
    top, bottom, left, right = dimensions
    outFile = frame[top:bottom, left:right]
    cv.imwrite(fileName, outFile)


if __name__ == '__main__':
    
    # Load details
    with open('config.json') as f:
        config_dict = json.load(f)

    # Load name of classes
    classes = config_dict['coco_classes']
    # Load links to ip cams
    ip_cams = config_dict['ip_cams']

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = "yolov3.cfg"
    modelWeights = "yolov3.weights"

    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # Process inputs
    cv.namedWindow("Stream", cv.WINDOW_NORMAL)
    cap = cv.VideoCapture(str(ip_cams[0]))
    if not cap.isOpened():
        raise IOError("Couldn't open webcam or video")

    while cv.waitKey(1) < 0:

        # Grab frame from video
        for _ in range(0, 24):
            cap.grab()
        hasFrame, frame = cap.read()

        # Crop the frame
        # (y_min, y_max) (x_min, x_max)
        # frame = frame[300:1080, 200:1920]
        
        # Stop the program if reached end of video
        if not hasFrame:
            cap.release()
            cv2.destroyAllWindows()
            exit()

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        # t, _ = net.getPerfProfile()
        # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        # print(label)

        cv.imshow("Stream", frame)