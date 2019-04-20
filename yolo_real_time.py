from yolo import YOLO
from timeit import default_timer as timer
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import numpy as np
from PIL import Image
import cProfile
import serial
import math
import colormath
from operator import add

def main(yolo):
    #ser = serial.Serial('/dev/ttyS0',baudrate=9600, timeout=3.0)
    pf=cProfile.Profile()
    pf.enable()
    vid = cv2.VideoCapture(0)
    grabbedShirt=0
    lockedRGB=[0,0,0]
    while True:
        for i in range(4):
            vid.grab()
        _, frame = vid.read()
        try:
            image = Image.fromarray(frame)
        except AttributeError as e:
            raise("Webcam isn't plugged in.")
        image, foundBoxes = yolo.detect_image(image)
        result = np.asarray(image)
        if yolo.draw:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pf.disable()
    pf.print_stats('cumtime')
    yolo.close_session()
    return 0


def angleCalc(box):
    #TODO
    return 0



if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--all_classes', default=False, action="store_true",help="Track all classes, not just people"
    )
    
    parser.add_argument(
        '--verbose','-v', type=int
    )
    parser.add_argument(
        '--draw',default=False,action="store_true",help="Draw image"
    )


    FLAGS = parser.parse_args()
    main(YOLO(**vars(FLAGS)))
