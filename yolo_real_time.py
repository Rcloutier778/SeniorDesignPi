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
from multiprocessing import Process, Queue


def serialHandler(ser,q):
    box=[0.0,0.0]
    while True:
        if not q.empty():
            box = q.get()
        distance,angle=angleCalc(box)
        if ser.in_waiting:
            serial_read = "Receoved " + ser.read(ser.in_waiting)
            #ser.write(serial_read.encode('UTF-8'))
            ser.write(str(distance))
            ser.write(0)
            ser.write(str(angle))
            ser.write(0)
            ser.reset_input_buffer()
        

def main(yolo):
    ser = serial.Serial('/dev/serial0',baudrate=9600, timeout=3.0)
    # Change the com to what the pi connects to the computer as
    pf=cProfile.Profile()
    pf.enable()
    com_ser.write("Ready\n")
    vid = cv2.VideoCapture(0)
    
    q = Queue()
    p = Process(target=serialHandler, args=(ser,q,))
    p.start()
    while True:
        for i in range(4):
            vid.grab()
        _, frame = vid.read()
        while True:
            try:
                image = Image.fromarray(frame)
                break
            except AttributeError as e:
                print("Webcam isn't plugged in?")
        
        image, foundBoxes = yolo.detect_image(image)
        
        #TODO: only grabbing first of the found boxes
        if foundBoxes:
            q.put(foundBoxes[0])
            
        if yolo.draw:
            result = np.asarray(image)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pf.disable()
    pf.print_stats('cumtime')
    p.join()
    yolo.close_session()
    return 0


def angleCalc(box):
    #TODO
    distance=0
    angle=0
    return distance,angle



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
