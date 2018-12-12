# JugglePro

Python and Arduino code needed to get the JugglePro system running.

## Getting Started

### Prerequisites

You will need the JugglePro system (hardware) and a laptop with a webcamera, Python, and Arduino installed.

### Instructions

1) Connect JugglePro system into laptop, and download haptics.ino onto the Arduino.
  a) To troubleshoot hardware connections, download and run multiplexer.ino onto the Arduino and ensure haptic controllers are detected on the Serial Monitor.
  b) If less than 2 I2C devices are detected, there is a loose connection in the hardware.
2) Run video.py to open visual system.
  a) If there's an error pertaining to the webcam, modify the cam_port on line 72 to the port of your webcam
  b) If there's an error pertaining to serial communication with the Arduino, modify the variables at the start of the main function starting at line 391.
3) Calibrate the computer vision algorithm to the color of the juggling balls by entering Calibration Mode by pressing the 'C' key on your computer.
  a) You will then be prompted three times to calibrate the color of each juggling ball where you slide the sliders on the 'tool' window.
  b) To properly calibrate, make sure that only the ball you are calibrating is being detected and no other ball is being detected.
  c) Press the spacebar to advance to the next ball color, press 'S' to save the calibration for future runs of the program, and press 'ESC' to exit the calibration prematurely.
4) Now that the system is fully plugged in and calibrated, you can put on the gloves and start tossing balls around and feel the adaptive vibrations of JugglePro!

Note: You do not need to follow the instructions displayed onscreen, but we suggest that you challenge yourself to get better.
