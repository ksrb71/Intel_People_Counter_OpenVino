import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
# device = "CPU"

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def draw_boxes(frame, result):
    '''
    Draw bounding boxes onto the frame.
    '''
    current_count = 0
    for box in result[0][0]:
        conf = box[2]
        if conf >= prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, "People_detected", (30, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)  # Put text when detected

            current_count = current_count + 1
    return frame, current_count


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU")

    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    return parser


def connect_mqtt():
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    global width, height, prob_threshold
    # Flag for the input image
    single_image_mode = False

    cur_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0

    # Initialise the class
    infer_network = Network()

    prob_threshold = args.prob_threshold

    model = args.model

    DEVICE = args.device
    CPU_EXTENSION = args.cpu_extension

    # Initialize the Inference Engine

    # Load the network model into the IE
    n, c, h, w = infer_network.load_model(model, args.device, 1, 1, cur_request_id, CPU_EXTENSION)[1]

    # Check if the input is a webcam

    if args.input == 'CAM':
        input_validated = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_validated = args.input

    # Checks for video file
    else:
        input_validated = args.input
        assert os.path.isfile(args.input), "file doesn't exist"

    cap = cv2.VideoCapture((input_validated))

    if (input_validated):
        cap.open(args.input)
    else:
        print("Not able to open input stream or file")

    prob_threshold = args.prob_threshold
    width = cap.get(3)
    height = cap.get(4)

    while cap.isOpened():

        # Reading the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        image = cv2.resize(frame, (w, h))

        # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))

        inf_start = time.time()

        # Perform inference on the frame
        infer_network.exec_net(cur_request_id, image)

        if infer_network.wait(cur_request_id) == 0:

            det_time = time.time() - inf_start

            result = infer_network.extract_output(cur_request_id)

            frame, current_count = draw_boxes(frame, result)

            inf_time_message = "Inference time: {:.3f}ms" \
                .format(det_time * 1000)

            cv2.putText(frame, inf_time_message, (10, 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

            # When new person enters the video
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))

            # Person duration in the video is calculated
            if current_count < last_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration", json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count

            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)

        # cap.release()
        # cv2.destroyAllWindows()
        # client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """

    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
