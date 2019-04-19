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
        if foundBoxes:
            if not grabbedShirt:
                box=foundBoxes[0]
                tracker=cv2.TrackerKCF_create()
                tracker.init(frame,tuple(box[0]-10,box[1]-10,box[2]-10,box[3]-10))
                '''
                for x in range(-10,10):
                    for y in range(-10,10):
                        lockedRGB=list(map(add, lockedRGB, frame[box[1]+y+box[3]//2][box[0]+x+box[2]//2]))
                #lockedRGB=frame[box[1]+box[3]//2][box[0]+box[2]//2]
                lockedRGB=[lockedRGB[0]//400, lockedRGB[1]//400, lockedRGB[2]//400]
                print(lockedRGB)
                '''
                
                grabbedShirt=1
                #TODO do moving avg for color 
            else:
                (success,box)=tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                for box in foundBoxes:
                    #draw box
                    cv2.rectangle(frame, (box[0]-10+box[2]//2, box[1]-10+box[3]//2), (box[0]+10+box[2]//2, box[1]+10+box[3]//2), (0,255,0),2)
                    '''
                    rgbval=[0,0,0]
                    for x in range(-10,10):
                        for y in range(-10,10):
                            rgbval=list(map(add, rgbval, frame[box[1]+y+box[3]//2][box[0]+x+box[2]//2]))
                    rgbval=[rgbval[0]//400, rgbval[1]//400, rgbval[2]//400]

                    distanceSquared = ((rgbval[0]-lockedRGB[0])**2 + (rgbval[1]-lockedRGB[1])**2 +(rgbval[2]-lockedRGB[2])**2)  
                    distance = math.sqrt(distanceSquared)
                    print('Distance between locked: ',lockedRGB, ' and ', rgbval,' is: ',distance)
                    '''
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
