#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Import the relevant files
from sys import exit as Die

try:
    import sys
    import time
    import cv2
    import time
    import imutils
    import numpy as np
    from threading import Thread
    from colordetection import ColorDetector
    import os
    import pickle
    import serial
except ImportError as err:
    Die(err)


class WebcamVideoStream:
    def __init__(self, src=0, name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


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
calibrationfile = "calibration.txt"
num_data_points = 30
time_limit = 30
RED = (0, 0, 200)
BLUE = (200, 0, 0)
GREEN = (0, 200, 0)


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
        defaultcal = {  # default color calibration (based on IDC garage)
            'red': [[179, 225, 160], [155, 160, 65]],
            'blue': [[105, 255, 180], [95, 200, 40]],
            'green': [[88, 192, 146], [30, 69, 73]]
        }
        file = open(calibrationfile, "wb")
        pickle.dump(defaultcal, file)
        file.close()

    colorcal = {}  # color calibration dictionary
    color = ['red', 'blue', 'green']  # list of valid colors
    line_color = [RED, BLUE, GREEN]

    cv2.resizeWindow('default', 1000, 1000)
    # create trackbars here
    cv2.createTrackbar('H Upper', "tool", defaultcal[color[len(colorcal)]][0][0], 179, empty_callback)
    cv2.createTrackbar('H Lower', "tool", defaultcal[color[len(colorcal)]][1][0], 179, empty_callback)
    cv2.createTrackbar('S Upper', "tool", defaultcal[color[len(colorcal)]][0][1], 255, empty_callback)
    cv2.createTrackbar('S Lower', "tool", defaultcal[color[len(colorcal)]][1][1], 255, empty_callback)
    cv2.createTrackbar('V Upper', "tool", defaultcal[color[len(colorcal)]][0][2], 255, empty_callback)
    cv2.createTrackbar('V Lower', "tool", defaultcal[color[len(colorcal)]][1][2], 255, empty_callback)

    cv2.createTrackbar('Max Ball Size', "tool", 100, 100, empty_callback)
    cv2.createTrackbar('Min Ball Size', "tool", 6, 100, empty_callback)

    # Remember that the range for S and V are not 0 to 179
    # make four more trackbars for ('S Upper', 'S Lower', 'V Upper', 'V Lower')
    # Note you should use these trackbar names to make other parts of the code run properly

    trajectory = [[] for _ in color]
    wait_for_peak = [True for _ in color]
    left_throw = False
    right_throw = False
    screen_width = 600
    limit_height = 200
    bottom_height = 400
    TURN_ON_LEFT = b'1'
    TURN_ON_RIGHT = b'2'

    # Actuation Variables
    checker_time = time.time()  # timer for determining actuation (checks drops and throws every 30 secs)
    throw_time = 0  # timer for time between throws
    drop_time = 0  # timer for time after ball(s) drop
    haptics_on = True
    msg = ''
    msg_frames = 200
    msg_frame_count = 0
    # There are six levels of actuation:
    # -  two for each level of juggling (1 ball, 2 balls, 3 balls)
    #   o  one with haptics and one without

    while True:
        frame = cam.read()
        while frame is None:
            pass
        frame = imutils.resize(frame, width=screen_width)  # may not be necessary
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # generates an hsv version of frame and

        # stores it in the hsv image variable
        key = cv2.waitKey(1) & 0xff

        # quit on escape.
        if key == 27:
            break

        # get area constraints
        max_size = cv2.getTrackbarPos('Max Ball Size', 'tool')
        min_size = cv2.getTrackbarPos('Min Ball Size', 'tool') + 1

        for index, name in enumerate(color):
            trajectory_color = line_color[index]
            # find current ball color
            if name == "red":
                # gets calibration values
                hu, su, vu = defaultcal[name][0]
                hl, sl, vl = defaultcal[name][1]
                # makes mask
                lower_hsv = np.array([0, sl, vl])
                upper_hsv = np.array([hl, su, vu])
                mask1 = cv2.inRange(hsv, lower_hsv, upper_hsv)
                lower_hsv = np.array([hu, sl, vl])
                upper_hsv = np.array([179, su, vu])
                mask2 = cv2.inRange(hsv, lower_hsv, upper_hsv)
                # make a mask with current ball color
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                upper_hsv = np.array(defaultcal[name][0])
                lower_hsv = np.array(defaultcal[name][1])
                # make a mask with current ball color
                mask = cv2.inRange(hsv, lower_hsv, upper_hsv)  # makes a mask where pixels with hsv in bounds

            # erosions and dilations remove any small blobs left in the mask
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # convert masked image to grayscale, run thresholding and contour detection
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]

            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if not (max_size > radius > min_size) or (y > bottom_height):
                    continue
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # draw some stuff!
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                # adds point to trajectory
                if len(trajectory[index]) == num_data_points:
                    trajectory[index] = trajectory[index][1:] + [center]
                else:
                    trajectory[index].append(center)

                end_of_parabola = False
                # draws trajectory up to penultimate point
                for i in range(1, len(trajectory[index]) - 1):
                    before = trajectory[index][i - 1]
                    after = trajectory[index][i]
                    if not (before[1] < limit_height) or not (after[1] < limit_height):
                        if end_of_parabola:
                            break
                        continue
                    end_of_parabola = True
                    # line_weight = int(np.sqrt(num_data_points / float(i + 1)) * 2)
                    cv2.line(frame,
                             before,
                             after,
                             trajectory_color,
                             2)

                # calculates drop time
                if throw_time > 0:
                    time_diff = time.time() - throw_time
                    if time_diff > 1:
                        drop_time += time_diff - 3
                throw_time = time.time()

                # last point in trajectory
                i = len(trajectory[index]) - 1
                before = trajectory[index][i - 1]
                after = trajectory[index][i]
                cv2.line(frame,
                         before,
                         after,
                         trajectory_color,
                         2)
                # checks if it is a left or right throw
                if after[0] > before[0]:  # if the ball is moving to the right in footage
                    left_throw = True
                elif after[0] < before[0]:  # if the ball is moving to the left in footage
                    right_throw = True
                # if the trajectory is going up and is near the top of the frame
                falling = (after[1] < before[1]) and (before[1] < limit_height)
                if not falling:
                    wait_for_peak[index] = True

                # TODO limit the following actions to only two and three-ball juggles
                if falling and wait_for_peak[index]:
                    if left_throw and haptics_on:
                        wait_for_peak[index] = False
                        left_throw = False
                        ser.write(TURN_ON_LEFT)
                        print("\nVibrate Left\n")
                    elif right_throw and haptics_on:
                        wait_for_peak[index] = False
                        right_throw = False
                        ser.write(TURN_ON_RIGHT)
                        print("\nVibrate Right\n")

        # draws a line showing the height one must throw the ball for detection
        cv2.line(frame,
                 (0, limit_height),
                 (screen_width, limit_height),
                 (0, 0, 0),
                 1)

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
                frame = cv2.flip(cam.read(), 1)
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

                # handles more orangish hint of red
                if color[len(colorcal)] == 'red':
                    lower_hsv = np.array([0, sl, vl])
                    upper_hsv = np.array([hl, su, vu])
                    mask1 = cv2.inRange(hsv, lower_hsv, upper_hsv)
                    lower_hsv = np.array([hu, sl, vl])
                    upper_hsv = np.array([179, su, vu])
                    mask2 = cv2.inRange(hsv, lower_hsv, upper_hsv)
                    # make a mask with current ball color
                    mask = cv2.bitwise_or(mask1, mask2)
                    lower_hsv = np.array([hl, sl, vl])
                    upper_hsv = np.array([hu, su, vu])
                else:
                    lower_hsv = np.array([hl, sl, vl])
                    upper_hsv = np.array([hu, su, vu])
                    # make a mask with current ball color
                    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)
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
                    throw_time = 0
                    drop_time = 0
                    checker_time = time.time()
                    break
                if key == 83:
                    # saves the calibration to a file
                    file = open(calibrationfile, "wb")
                    pickle.dump(defaultcal, file)
                    file.close()
                    print("file saved")

        if time.time() - checker_time > time_limit:  # 30 seconds have passed since last check
            if drop_time / time_limit <= .1 and throw_time != 0:
                if not haptics_on:
                    msg = "You may now juggle one more ball"
                haptics_on = not haptics_on  # swaps between on and off when upgrading
            elif drop_time / time_limit >= .5 or throw_time == 0:
                if not haptics_on:
                    haptics_on = True
                msg = "You should juggle one less ball"
            throw_time = 0
            drop_time = 0
            checker_time = time.time()
        # show result
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, msg, (150, 150), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("default", frame)
        if msg != '':  # remove actuation message
            msg_frame_count += 1
            if msg_frame_count >= msg_frames:
                msg_frame_count = 0
                msg = ''
    # end of scan()
    cv2.destroyAllWindows()
    cam.stop()


if __name__ == '__main__':
    port = 'COM3'  # change based on Arduino
    baudrate = 115200  # change based on Arduino
    ser = serial.Serial(port, baudrate, writeTimeout=0)
    time.sleep(2)
    cv2.namedWindow('default', cv2.WINDOW_NORMAL)
    cv2.namedWindow('tool', cv2.WINDOW_NORMAL)
    scan()
    ser.close()
