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

FOCAL_LEN = 4.8589
PREV_SIDE = 0
                

def main(yolo):
    ser = serial.Serial('/dev/serial0',baudrate=9600, timeout=3.0)

    # Change the com to what the pi connects to the computer as
    pf=cProfile.Profile()
    pf.enable()
    vid = cv2.VideoCapture(0)
    prevUsed=0
    distance,angle=None,None
    print("running")
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
        if ser.in_waiting:
            ser.reset_input_buffer()
            if foundBoxes:
                distance,angle=angleCalc0(foundBoxes, image.width, image.height)
            else:
                prevUsed+=1
                if angle != None:
                    if angle > 0.0:
                        angle +=10
                    else:
                        angle -= 10
                if prevUsed > 3:
                    distance,angle=None, None
                    prevUsed=0
                
            print("Sending to K64")
            print("Distance",str(distance))
            print("Angle",str(angle))
            if distance==None:
                ser.write(chr(0x01).encode('ASCII'))
            else:
                ser.write(str(distance).encode('ASCII'))
            ser.write(chr(0x00).encode('ASCII'))
            if angle==None:
                ser.write(chr(0x01).encode('ASCII'))
            else:
                ser.write(str(angle).encode('ASCII'))
            ser.write(chr(0x00).encode('ASCII'))
            
        if yolo.draw:
            result = np.asarray(image)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pf.disable()
    pf.print_stats('cumtime')
    yolo.close_session()
    return 0
    

def rc_angleCalc(box, iw):
    x,y,w,h=box
    userCenter=x+w//2
    cboxSize=150
    centerBox=[(iw-cboxSize)//2, cboxSize+((iw-cboxSize)//2)]
    if userCenter < centerBox[0]: #left
        #[-90,-10]
        return -(150 - (userCenter * 140 / centerBox[0]))
    elif userCenter > centerBox[1]: #right
        #[10,90]
        return 70 + ((userCenter-centerBox[1]) * 140 // (centerBox[0]))
    else:
        return 0.0

    
def angleCalc0(box, iw, ih, th=72):
    # box = box_gen(x, y, h, w)
    # dist = get_target_dist_f([h, th], [w, tw], m)
    x=box[0]
    h=box[3]
    distance = FOCAL_LEN * 6.0 * ih / (h*55.0/12.0)
    if h >= (0.8*ih)-10 or box[2] >= iw/3:
        distance=0
    return [distance, rc_angleCalc(box,iw)]


"""
Calcs a coarse turn angle based on size of the 
target relative to the image

If the target dissappears from view between two frames,
PREV_SIDE is used to calculate the correction angle

Parameters:
x (int): the x coord of the origin for the target
h (int): the height of the target box
image_w (int): the width of the image

Returns:
int: Right turn (+), Left turn (-), No turn (0)
"""
def angleCalc(x, h, image_w):
    r = (image_w / 2)
    side = angleCalc1(x, image_w)

    if not side[1]:
        ##side = side[0]
        h = 1
        r = 0

    if h <= r * 0.05:
        return side[0] * 1
    elif h <= r * 0.25:
        return side[0] * 10
    elif h <= r * 0.7:
        return side[0] * 30
    else:
        return side[0] * 45
        
        
"""
Which side of center is target on

Parameters:
x (int): the x coord of the origin for the target
image_w (int): the width of the image

Returns:
int: Right (1), Left (-1), or Center (0)

"""
def angleCalc1(x, image_w):
    half = image_w / 2
    global PREV_SIDE

    if half < x:
        PREV_SIDE = 1
        return 1, True
    elif half > x:
        PREV_SIDE = -1
        return -1, True
    elif half == x:
        PREV_SIDE = 0
        return 0, True
    elif x == None:
        return PREV_SIDE, False
        

def get_target_dist(h, th, w=0, tw=0, method=1):
    if method == 1:
        return (h * FOCAL_LEN) / th

    elif method == 2:
        return (w * FOCAL_LEN) / tw

    elif method == 3:
        return ((h * FOCAL_LEN) / th + w * FOCAL_LEN / tw) / 2



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
