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
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ # 16Hz max

mlx_shape = (24,32)
tdata = np.zeros((24*32,))
t_img = (np.reshape(tdata,mlx_shape))
alpha = 0.5

def temp2Que(tempQueue):
    while True:
        mlx.getFrame(tdata) # read MLX temperatures into frame var
        t_img = (np.reshape(tdata,mlx_shape)) # reshape to 24x32 print(t_img.shape) => (24, 32)
        tempQueue.put(t_img)

def classify_frame(net, inputQueue, outputQueue):
    # keep looping
    while True:
        # check to see if there is a frame in our input queue
        if not inputQueue.empty():
            # grab the frame from the input queue, resize it, and
            # construct a blob from it
            frame = inputQueue.get()
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            
            # set the blob as input to our deep learning object
            # detector and obtain the detections
            net.setInput(blob)
            detections = net.forward()
            
            # write the detections to the output queue
            outputQueue.put(detections)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
net_confidence = 0.5
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# initialize multiprocessing
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None
tempQueue = Queue(maxsize=1)

print("[INFO] starting face detection process...")
p0 = Process(target=classify_frame, args=(net, inputQueue, outputQueue,))
p0.daemon = True
p0.start()

print("[INFO] starting thermal detection process...")
p1 = Process(target=temp2Que, args=(tempQueue,))
p1.daemon = True
p1.start()

print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(2.0) # allow the camera sensor to warm up for 2 seconds
fps = FPS().start()

# loop over the frames from the video stream
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    (h, w) = frame.shape[:2]
    if inputQueue.empty(): inputQueue.put(frame)
    if not outputQueue.empty(): detections = outputQueue.get()
    if not tempQueue.empty(): t_img = tempQueue.get()
    
    # loop over the detections
    if (detections is not None) and (t_img is not None):
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if (confidence < net_confidence) or \
               (detections[0, 0, i, 3:7].max()>1) or \
               (detections[0, 0, i, 3] > detections[0, 0, i, 5]) or \
               (detections[0, 0, i, 4] > detections[0, 0, i, 6]): 
               continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            tbox = detections[0, 0, i, 3:7] * np.array([32, 24, 32, 24])
            (startTX, startTY, endTX, endTY) = tbox.astype("int")
            if startTX>0 : startTX -= 1
            if startTY>0 : startTY -= 1          
            if endTX <32 : endTX += 1
            if endTY <24 : endTY += 1
            tmax = t_img[startTY:endTY, startTX:endTX].max()
            #text = "Tmax={:.1f} C".format(tmax)
            text = "Tmax={:.1f} F".format(tmax*9/5 + 32)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    # show the output frame
    cv2.imshow('Face Temperature', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()


