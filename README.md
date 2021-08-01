# ThermalFaceDetection
In this project, I will be using the MLX90640 Far infrared thermal sensor array (110° FOV, 32x24 RES) and a 110° FOV camera compatible with the Nvidia Jetson Nano to build a thermal face detection device.

## [1] MLX90640 Setup

### [1-1] Install Jetson.GPIO
https://github.com/NVIDIA/jetson-gpio
```
$ sudo pip3 install Jetson.GPIO
$ sudo apt-get install -y libi2c-dev i2c-tools
```
You can use the following command to check I2C bus device
```
$ sudo i2cdetect -y -r 1
```
result:
```
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:          -- -- -- -- -- -- -- -- -- -- -- -- -- 
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
30: -- -- -- 33 -- -- -- -- -- -- -- -- -- -- -- -- 
40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
```

### [1-2] Install Adafruit Library
```
$ sudo apt-get install -y python-smbus
$ sudo pip3 install adafruit-blinka
$ sudo pip3 install adafruit-circuitpython-mlx90640
```

### [1-3] Connections between MLX90640 and the Jetson Nano

Pin connections between the Jetson Nano and MLX90640

![alt text](https://github.com/xyth0rn/ThermalFaceDetection/blob/main/nano_mlx.png)

Circuit recommended by Melexis

![alt text](https://github.com/xyth0rn/ThermalFaceDetection/blob/main/mlx90640_cir.png)

Jeson Nano GPIO map

![alt text](https://github.com/xyth0rn/ThermalFaceDetection/blob/main/Jetson_Nano_GPIO.png)

### [1-4] Use EasyEDA & CNC to Create a Simple Circuit Board (optional)
I designed the schematics with EasyEDA, then used a simple CNC to make the circuit board.
This step is optional if CNC is not available and can be replaced with using a breadboard or strip board.

Schematics
![alt text](https://github.com/xyth0rn/ThermalFaceDetection/blob/main/mlx90640_sch.png)

Layout
![alt text](https://github.com/xyth0rn/ThermalFaceDetection/blob/main/mlx90640_lay.png)

CNC
![alt text](https://github.com/xyth0rn/ThermalFaceDetection/blob/main/pcb_1.jpeg)
![alt text](https://github.com/xyth0rn/ThermalFaceDetection/blob/main/pcb_2.jpeg)
![alt text](https://github.com/xyth0rn/ThermalFaceDetection/blob/main/pcb_3.jpeg)

### [1-5] Assembling MLX90640 FLIR, camera, and the Jetson Nano
![alt text](https://github.com/xyth0rn/ThermalFaceDetection/blob/main/pcb_4.jpeg)
![alt text](https://github.com/xyth0rn/ThermalFaceDetection/blob/main/pcb_5.jpeg)

Since I am using a 110° FOV version MLX90640, the camera should also be 110° FOV.

*Note: There are 2 versions of MLX90640, being the 55° FOV version and the 110° FOV version.*

## [2] Python Programs (Python3 only)

### [2-1] 1_mlx90640.py

![alt text]()

### [2-2] 2_cam_overlay_mlx.py
If the FOV of your camera is not perfect match with the FOV of MLX90640, you can use this program to check the difference of FOV. 
![alt text]()

### [2-3] 3_thermalFaceDetection_1.py
the simplest version, use jetson.inference & facenet face detection.
the resulotion of jetson.utils.videoSource("csi://0") is 16:9 (PS: 4:3 resolution is too high)
the resulotion of MLX90640 is 4:3
.....
![alt text]()

### [2-4] 4_thermalFaceDetection_2.py
"This model is based on Single-Shot-Multibox detector and uses ResNet-10 Architecture as backbone."
use cv2.dnn.readNetFromCaffe.
![alt text]()

## Youtube link

## Reference
Melexis MLX90640:
https://www.melexis.com/en/product/mlx90640/far-infrared-thermal-sensor-array
https://www.reddit.com/r/JetsonNano/comments/jkrjye/mlx90640_32x24_interpolated_to_640x480_on_the/

Hello AI World: 
https://github.com/dusty-nv/jetson-inference#hello-ai-world

LearnOpenCV: Face Detection – OpenCV, Dlib and Deep Learning ( C++ / Python )
https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
https://www.pyimagesearch.com/2017/10/16/raspberry-pi-deep-learning-object-detection-with-opencv/

pyimagesearch: Face detection with OpenCV and deep learning
https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/


