# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils
import time
import cv2

import time,board,busio
import adafruit_mlx90640
import datetime as dt

i2c = busio.I2C(board.SCL, board.SDA, frequency=400000) # setup I2C
mlx = adafruit_mlx90640.MLX90640(i2c) # begin MLX90640 with I2C comm
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ # 16Hz max

mlx_shape = (24,32)
tdata = np.zeros((24*32,))
alpha = 0.5
tframe = np.reshape(np.zeros((480*640,)), (480,640))

def td_to_img(f,tmax,tmin):
    norm = np.uint8((f - tmin)*255/(tmax-tmin))
    return norm

def tframe2Que(outputQueue):
    while True:
        mlx.getFrame(tdata) # read MLX temperatures into frame var
        t_img = (np.reshape(tdata,mlx_shape)) # reshape to 24x32
        tmax = tdata.max()
        tmin = tdata.min()
        ta_img = td_to_img(t_img, tmax, tmin)
        # Image processing
        img = cv2.applyColorMap(ta_img, cv2.COLORMAP_JET) # cv2.COLORMAP_HSV
        img = cv2.resize(img, (640,480), interpolation = cv2.INTER_CUBIC)
        # img = cv2.flip(img, 1)
        outputQueue.put(img)

print("[INFO] starting process...")
outputQueue = Queue(maxsize=1)
p = Process(target=tframe2Que, args=(outputQueue,))
p.daemon = True
p.start()

print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(2.0) # allow the camera sensor to warm up for 2 seconds

# loop over the frames from the video stream
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # if the output queue *is not* empty, grab the detections
    if not outputQueue.empty():
        tframe = outputQueue.get()
    if len(tframe.shape) < 3: continue
    # show the output frame
    cv2.addWeighted(frame, alpha, tframe, 1-alpha, 0, frame)
    cv2.imshow('FaceTemp', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# stop the timer and display FPS information
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
