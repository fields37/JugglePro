#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Import the relevant files
from sys import exit as Die
try:
    import sys
    import cv2
    import numpy as np
    from colordetection import ColorDetector
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
cam_port         = 1 # your code here task 1.1
cam              = cv2.VideoCapture(cam_port) # your code here task 1.1



"""
    We are going to want to draw some stickers. In the solution
    we have 3 sets of stickers. Each set has 9 stickers to represent
    each sticker on any given face of the rubik's cube.

    The 'stickers' set of stickers are physical markers used to help the
    user see where to place the cube for color scanning.

    The 'current_stickers' set show which colors are currently being detected 
    for each position on the face.

    The 'preview_stickers' set shows the most recently recorded face.
    You can rescan a face if the preview does not match the actual colors
    on the face of the rubik's cube.

    These are the coordinates of each sticker for each set. Feel free to play
    with these values if you don't like the sticker placement.

    *note shorten split
"""
detector_stickers = [[200, 120], [300, 120], [400, 120],
                   [200, 220], [300, 220], [400, 220],
                   [200, 320], [300, 320], [400, 320]]

current_stickers = [[20, 20], [54, 20], [88, 20],
                   [20, 54], [54, 54], [88, 54],
                   [20, 88], [54, 88], [88, 88]]

recorded_stickers = [[20, 130], [54, 130], [88, 130],
                   [20, 164], [54, 164], [88, 164],
                   [20, 198], [54, 198], [88, 198]]



def color_to_notation(color):
    """
    Helper function for converting colors to notation
    used by solver.
    """
    notation = {
        'green'  : 'F',
        'white'  : 'U',
        'blue'   : 'B',
        'red'    : 'R',
        'orange' : 'L',
        'yellow' : 'D'
    }
    return notation[color]

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

    sides   = {}                            # collection of scanned sides
    preview = ['white','white','white',     # default starting preview sticker colors
               'white','white','white',
               'white','white','white']
    state   = [0,0,0,                       # current sticker colors
               0,0,0,
               0,0,0]

    defaultcal = {                          # default color calibration
                'ball1':[[34,255,255],[17,0,0]],
                }

    colorcal  = {}                          # color calibration dictionary
    color = ['ball1']  # list of valid colors            
    
    cv2.namedWindow('default',0)
    # create trackbars here
    cv2.createTrackbar('H Upper',"default",defaultcal[color[len(colorcal)]][0][0],179, empty_callback)
    cv2.createTrackbar('H Lower',"default",defaultcal[color[len(colorcal)]][1][0],179, empty_callback)
    cv2.createTrackbar('S Upper',"default",defaultcal[color[len(colorcal)]][0][1],255, empty_callback)
    cv2.createTrackbar('S Lower',"default",defaultcal[color[len(colorcal)]][1][1],255, empty_callback)
    cv2.createTrackbar('V Upper',"default",defaultcal[color[len(colorcal)]][0][2],255, empty_callback)
    cv2.createTrackbar('V Lower',"default",defaultcal[color[len(colorcal)]][1][2],255, empty_callback)

    cv2.createTrackbar('Max Ball Size',"default",2000,2000, empty_callback)
    cv2.createTrackbar('Min Ball Size',"default",500,2000, empty_callback)

    # Remember that the range for S and V are not 0 to 179
    # make four more trackbars for ('S Upper', 'S Lower', 'V Upper', 'V Lower')
    # Note you should use these trackbar names to make other parts of the code run properly


    colorcal = defaultcal

    ##################################################
    # Task 1: you can insert out of the loop code here
    ##################################################


    while cameratesting:
        '''
        Here we want to make sure things are working and learn about how to use some openCV functions
        Your code here
        '''
        #task 1.2 preview a camera window
        # Read a frame and display it in the window

        # Captures a frame of video from the camera object
        _,frame = cam.read()

        #################################
        #Add more processing code here
        #################################

        # Convert frame from RGB to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # generates an hsv version of frame and 
                                                     # stores it in the hsv image variable
        
        # create a mask
        # Bounds for HSV values we are interested in (Blue)
        lower_hsv = np.array([89,178,51])     #hmin,smin,vmin
        upper_hsv = np.array([118,255,194])   #hmax,smax,vmax

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv) # makes a mask where pixels with hsv in bounds

        # Apply the mask and display
        # Bitwise and the frame with itself and apply the mask
        frame = cv2.bitwise_and(frame,frame, mask= mask)

        # Draw rectangle on the frame
        cv2.rectangle(frame, (200,200), (250, 250), (255,0,0), 2) 

        # -1 borderwidth is a fill
        cv2.rectangle(frame, (300,200), (350, 250), (0,0,255), -1)

        # arg1 = trackbar name
        # arg2 = window to pull trackbar info from
        value = cv2.getTrackbarPos('My track bar','my_window_name')
        print(value)

        # Note the construction of a rectangle
        # arg1 = frame to draw on
        # arg2 = x,y coordinates of the rectangle's top left corner
        # arg3 = x,y coordinates of the rectangle's bottom right corner
        # arg4 = r,g,b values
        # arg5 = borderwidth  => width of the border or make a fill using -1

        # cv2.rectangle(frame, (xtop_left,ytop_left), (xbot_right,ybot_right), (r,g,b), borderwidth)

        # Displays the frame on the window we made
        cv2.imshow('my_window_name', frame) 

        # Sets the amount of time to display a frame in milliseconds
        key = cv2.waitKey(10)

        #task 1.3 draw a rectangle
        #task 1.4 make a slider
        #task 1.5 add text
        #task 1.6 make a mask based on hsv
        #task 1.7 display the masked image


    while not cameratesting:
        _, frame = cam.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # generates an hsv version of frame and 
                                                     # stores it in the hsv image variable
        key = cv2.waitKey(10)


        for index,(x,y) in enumerate(detector_stickers):
            roi          = hsv[y:y+32, x:x+32]              # extracts hsv values within sticker
            avg_hsv      = ColorDetector.median_hsv(roi)    # filters the hsv values into one hsv
            color_name   = ColorDetector.get_color_name(avg_hsv,colorcal) # extracts the color based on hsv
            state[index] = color_name                       # stores the color 

            # update when space bar is pressed.
            if key == 32:
                preview = list(state)
                face = color_to_notation(state[4])          # convert the color to notation of the middle sticker and label this as the face
                notation = [color_to_notation(color) for color in state] # convert all colors to notation
                sides[face] = notation                      # update the face in the sides dictionary


        # quit on escape.
        if key == 27:
            break


        # find current ball color
        upper_hsv = np.array(defaultcal['ball1'][0])
        lower_hsv = np.array(defaultcal['ball1'][1])

        # make a mask with current ball color
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv) # makes a mask where pixels with hsv in bounds
        res = cv2.bitwise_and(frame,frame, mask= mask)

        # convert masked image to grayscale, run thresholding and contour detection
        imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        reset, thresh = cv2.threshold(imgray, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # get area constraints
        max_size = cv2.getTrackbarPos('Max Ball Size','default')
        min_size = cv2.getTrackbarPos('Min Ball Size','default')

        culled_contours = []
        for contour in contours:
            if max_size > len(contour) > min_size:
                culled_contours.append(contour)

        

        # draw some ellipses and contours
        for i in range(len(culled_contours)):
            contour = culled_contours[i]

            # fit that ellipse!
            shape = cv2.fitEllipse(contour)

            # draw some stuff!
            cv2.drawContours(frame, culled_contours, i, (0,255,0), 3)
            cv2.ellipse(frame, shape, (0,255,255), 2, 8)



        # show result
        cv2.imshow("default", frame)



        if key == 99: 
            colorcal = {}   
            while len(colorcal) < len(defaultcal):
                _, frame = cam.read()
                
                
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                key = cv2.waitKey(10) & 0xff

                # hue upper lower
                hu = cv2.getTrackbarPos('H Upper','default')
                hl = cv2.getTrackbarPos('H Lower','default')
                # saturation upper lower
                su = cv2.getTrackbarPos('S Upper','default')
                sl = cv2.getTrackbarPos('S Lower','default')
                # value upper lower
                vu = cv2.getTrackbarPos('V Upper','default')
                vl = cv2.getTrackbarPos('V Lower','default')

                lower_hsv = np.array([hl,sl,vl])
                upper_hsv = np.array([hu,su,vu])
                
                # Task 3
                mask = cv2.inRange(hsv, lower_hsv, upper_hsv) # makes a mask where pixels with hsv in bounds
                res = cv2.bitwise_and(frame,frame, mask= mask)
                
                if key == 32:
                    defaultcal[color[len(colorcal)]] = [upper_hsv,lower_hsv]
                    colorcal[color[len(colorcal)]] = [upper_hsv,lower_hsv]

                    if(len(colorcal) < len(defaultcal)):
                        cv2.setTrackbarPos('H Upper','default',defaultcal[color[len(colorcal)]][0][0])
                        cv2.setTrackbarPos('S Upper','default',defaultcal[color[len(colorcal)]][0][1])
                        cv2.setTrackbarPos('V Upper','default',defaultcal[color[len(colorcal)]][0][2])
                        cv2.setTrackbarPos('H Lower','default',defaultcal[color[len(colorcal)]][1][0])
                        cv2.setTrackbarPos('S Lower','default',defaultcal[color[len(colorcal)]][1][1])
                        cv2.setTrackbarPos('V Lower','default',defaultcal[color[len(colorcal)]][1][2])

                if(len(colorcal) < len(defaultcal)):
                    text = 'calibrating {}'.format(color[len(colorcal)])
                cv2.putText(res, text, (20, 460), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                cv2.imshow("default", res)
                # quit on escape key.
                if key == 27:
                    break

    cam.release()
    cv2.destroyAllWindows()
    return sides if len(sides) == 6 else False


