#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Import the relevant files
from sys import exit as Die

try:
    import sys
    import cv2
    import time
    import imutils
    import numpy as np
    from imutils.video import WebcamVideoStream
    from imutils.video import FPS
    from colordetection import ColorDetector
    import os
    import pickle
except ImportError as err:
    Die(err)

'''
Testing variable
set to false to start task 2 & 3
'''
cameratesting = False

'''
Initialize the camera here
'''
cam_port = 0
cam = WebcamVideoStream(src=cam_port).start()
fps = FPS().start()
calibrationfile = "calibration.txt"
num_data_points = 15


def empty_callback(x):
    '''
    Empty function for callback when slider positions change. Need input x, this is the value 
    the slider has changed to. You don't need to do anything in this function.
    '''
    pass


def scan():
    """
    Open up the webcam and scans the 9 regions in the center
    and show a preview.

    After hitting the space bar to confirm, the block below the
    current stickers shows the current state that you have.
    This is show every user can see what the computer took as input.

    :returns: dictionary
    """
    # Read the calibration values from file
    if os.path.exists(calibrationfile):
        file = open(calibrationfile, "rb")
        defaultcal = pickle.load(file)
        file.close()
    # If no calibration file exists, create it and use the default values
    else:
        defaultcal = {  # default color calibration
            'ball1': [[175, 230, 255], [155, 130, 100]],
            'ball2': [[102, 255, 255], [94, 75, 130]],
            'ball3': [[94, 255, 254], [75, 135, 0]]
        }
        file = open(calibrationfile, "wb")
        pickle.dump(defaultcal, file)
        file.close()

    colorcal = {}  # color calibration dictionary
    color = ['ball1', 'ball2', 'ball3']  # list of valid colors
    line_color = ['']

    cv2.resizeWindow('default', 1000, 1000)
    # create trackbars here
    cv2.createTrackbar('H Upper', "tool", defaultcal[color[len(colorcal)]][0][0], 179, empty_callback)
    cv2.createTrackbar('H Lower', "tool", defaultcal[color[len(colorcal)]][1][0], 179, empty_callback)
    cv2.createTrackbar('S Upper', "tool", defaultcal[color[len(colorcal)]][0][1], 255, empty_callback)
    cv2.createTrackbar('S Lower', "tool", defaultcal[color[len(colorcal)]][1][1], 255, empty_callback)
    cv2.createTrackbar('V Upper', "tool", defaultcal[color[len(colorcal)]][0][2], 255, empty_callback)
    cv2.createTrackbar('V Lower', "tool", defaultcal[color[len(colorcal)]][1][2], 255, empty_callback)

    cv2.createTrackbar('Max Ball Size', "tool", 2000, 2000, empty_callback)
    cv2.createTrackbar('Min Ball Size', "tool", 50, 2000, empty_callback)

    # Remember that the range for S and V are not 0 to 179
    # make four more trackbars for ('S Upper', 'S Lower', 'V Upper', 'V Lower')
    # Note you should use these trackbar names to make other parts of the code run properly

    trajectory = [[] for _ in color]

    while True:
        frame = cam.read()
        while frame is None:
            pass
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # generates an hsv version of frame and
        # stores it in the hsv image variable
        key = cv2.waitKey(1) & 0xff

        # quit on escape.
        if key == 27:
            break

        # get area constraints
        max_size = cv2.getTrackbarPos('Max Ball Size', 'tool')
        min_size = cv2.getTrackbarPos('Min Ball Size', 'tool') + 5

        for name in color:
            trajectory_index = color.index(name)
            # find current ball color
            upper_hsv = np.array(defaultcal[name][0])
            lower_hsv = np.array(defaultcal[name][1])

            # make a mask with current ball color
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)  # makes a mask where pixels with hsv in bounds
            res = cv2.bitwise_and(frame, frame, mask=mask)

            # convert masked image to grayscale, run thresholding and contour detection
            imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            reset, thresh = cv2.threshold(imgray, 60, 255, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            max_contour = None
            max_contour_size = 0
            for contour in contours:
                if max_size > len(contour) > min_size:
                    if len(contour) > max_contour_size:
                        max_contour = contour
                        max_contour_size = len(contour)

            # fit that ellipse!
            if max_contour is not None:
                shape = cv2.fitEllipse(max_contour)

                # draw some stuff!
                cv2.drawContours(frame, [max_contour], 0, (0, 255, 0), 3)
                cv2.ellipse(frame, shape, (0, 255, 255), 2, 8)
                point = int(shape[0][0]), int(shape[0][1])
                if len(trajectory[trajectory_index]) == num_data_points:
                    trajectory[trajectory_index] = trajectory[trajectory_index][1:] + [point]
                else:
                    trajectory[trajectory_index].append(point)

                # draws trajectory
                for i in range(len(trajectory[trajectory_index]) - 1):
                    cv2.line(frame,
                             trajectory[trajectory_index][i],
                             trajectory[trajectory_index][i + 1],
                             (255, 255, 255),
                             3)

        # show result
        cv2.imshow("default", frame)
        fps.update()

        # If "c" is pressed, enter calibration sequence
        if key == 99:
            colorcal = {}
            cv2.setTrackbarPos('H Upper', 'tool', defaultcal[color[len(colorcal)]][0][0])
            cv2.setTrackbarPos('S Upper', 'tool', defaultcal[color[len(colorcal)]][0][1])
            cv2.setTrackbarPos('V Upper', 'tool', defaultcal[color[len(colorcal)]][0][2])
            cv2.setTrackbarPos('H Lower', 'tool', defaultcal[color[len(colorcal)]][1][0])
            cv2.setTrackbarPos('S Lower', 'tool', defaultcal[color[len(colorcal)]][1][1])
            cv2.setTrackbarPos('V Lower', 'tool', defaultcal[color[len(colorcal)]][1][2])

            while len(colorcal) < len(defaultcal):
                frame = cam.read()

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                key = cv2.waitKey(1) & 0xff

                # hue upper lower
                hu = cv2.getTrackbarPos('H Upper', 'tool')
                hl = cv2.getTrackbarPos('H Lower', 'tool')
                # saturation upper lower
                su = cv2.getTrackbarPos('S Upper', 'tool')
                sl = cv2.getTrackbarPos('S Lower', 'tool')
                # value upper lower
                vu = cv2.getTrackbarPos('V Upper', 'tool')
                vl = cv2.getTrackbarPos('V Lower', 'tool')

                lower_hsv = np.array([hl, sl, vl])
                upper_hsv = np.array([hu, su, vu])

                # Task 3
                mask = cv2.inRange(hsv, lower_hsv, upper_hsv)  # makes a mask where pixels with hsv in bounds
                res = cv2.bitwise_and(frame, frame, mask=mask)

                if key == 32:
                    defaultcal[color[len(colorcal)]] = [upper_hsv, lower_hsv]
                    colorcal[color[len(colorcal)]] = [upper_hsv, lower_hsv]

                    if len(colorcal) < len(defaultcal):
                        cv2.setTrackbarPos('H Upper', 'tool', defaultcal[color[len(colorcal)]][0][0])
                        cv2.setTrackbarPos('S Upper', 'tool', defaultcal[color[len(colorcal)]][0][1])
                        cv2.setTrackbarPos('V Upper', 'tool', defaultcal[color[len(colorcal)]][0][2])
                        cv2.setTrackbarPos('H Lower', 'tool', defaultcal[color[len(colorcal)]][1][0])
                        cv2.setTrackbarPos('S Lower', 'tool', defaultcal[color[len(colorcal)]][1][1])
                        cv2.setTrackbarPos('V Lower', 'tool', defaultcal[color[len(colorcal)]][1][2])

                if len(colorcal) < len(defaultcal):
                    text = 'calibrating {}'.format(color[len(colorcal)])
                cv2.putText(res, text, (20, 460), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow("default", res)
                # quit on escape key.
                if key == 27:
                    break
            # saves the calibration to a file
            file = open(calibrationfile, "wb")
            pickle.dump(defaultcal, file)
            file.close()
            print("file saved")

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    cam.stop()


if __name__ == '__main__':
    cv2.namedWindow('default', cv2.WINDOW_NORMAL)
    cv2.namedWindow('tool', cv2.WINDOW_NORMAL)
    scan()
