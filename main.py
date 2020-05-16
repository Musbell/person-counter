"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60




def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
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
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def main():
    """
    Load the network and parse the SSD output.
    :return: None
    """
    # Connect to the MQTT server
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    args = build_argparser().parse_args()

    total_count = 0
    last_count = 0
    start_time = 0
    request_id = 0

    # Initialize the Inference Engine
    infer_network = Network()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    infer_network.load_model(args.model, args.device, num_requests=0)
    n, c, h, w = infer_network.get_input_shape()

    if args.input == "CAM":
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    try:
        cap = cv2.VideoCapture(args.input)
    except FileNotFoundError:
        print("Cannot locate video file: " + args.input)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    if input_stream:
        cap.open(args.input)
    if not cap.isOpened():
        log.error("Can't to open video source")
    prob_threshold = args.prob_threshold
    cap_w = cap.get(3)
    cap_h = cap.get(4)

    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        img = cv2.resize(frame, (w, h))
        img = img.transpose((2, 0, 1))
        img = img.reshape((n, c, h, w))
        inf_start = time.time()
        infer_network.exec_net(img, request_id=0)

        if infer_network.wait(request_id) == 0:
            det_time = time.time() - inf_start
            result = infer_network.get_output(request_id)

            current_count = 0
            for obj in result[0][0]:
                # Draw bounding box for object when it's probability is more than
                #  the specified threshold
                if obj[2] > prob_threshold:
                    xmin = int(obj[3] * cap_w)
                    ymin = int(obj[4] * cap_h)
                    xmax = int(obj[5] * cap_w)
                    ymax = int(obj[6] * cap_h)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (225, 225, 225), 1)
                    current_count = current_count + 1
            inf_time_message = "Inference time: {:.3f}ms" \
                .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))

            if current_count < last_count:
                duration = int(time.time() - start_time)
                client.publish("person/duration",
                               json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count

            if key_pressed == 27:
                break
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()

if __name__ == '__main__':
    main()
    exit(0)
