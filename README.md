# ESP32Cam-ObjectCounting

## Project description
This project aims to create a real-time object counting application using an ESP32-CAM module to capture images and a server (PC) for object detection using the YOLOv3 algorithm with OpenCV in Python. 
- **main branch**: All objects listed in the coco.names file are detected and the total count is displayed.
- **feature/lydia/specific-detection branch**: The detected objects include cars, trucks and buses, which are specified in the YOLO COCO dataset. Many approaches are involved as follow:
    - counter_native.py: the yolov3 model runs on each image sent from the esp32 cam and counts every object (in case of multiple images sent to simulate a video: it doesn't take into account the fact that the same object can be counted multiple times i.e none tracked object)
    - counter_deep_sort_video.py: the yolov7 model runs on a local video (.mp4) to detect, track and count each object using the deep sort algorithm implemented by ikomia (the best implementation)
    - counter_deep_sort_espCam.py: the yolov7 model runs on multiple images sent consecutively from the esp32 cam using deep sort algorithm (not promising performance)

## Project Components:
1. ESP32-CAM Code

    The ESP32-CAM captures images at different resolutions (low, medium, high) and sends them to the server.
    Images are accessible through HTTP endpoints (/cam-lo.jpg, /cam-hi.jpg, /cam-mid.jpg).

2. Server Code

    The server retrieves images from the ESP32-CAM and performs object detection on these images, specifically counting cars, trucks and buses.
    The count is displayed in real-time on the server's graphical user interface.

## Requirements:

    - ESP32-CAM module (AI Thinker model) with appropriate connections, I suggest you to test the camera using `test_live_cam.py`
    - Python environment installed on the server (run `pip install -r requirements.txt`)
    - Download the YOLOv3-320 weights files from darknet [website](https://pjreddie.com/darknet/yolo/).

## Setup:

    - Configure ESP32-CAM code with appropriate Wi-Fi credentials in the file `capture-esp32cam.ino`
    - Upload the ESP32-CAM code on the module using Arduino IDE.
    - Grap the ip address printed and update the url variabe in the server code. 
    - Ensure the server has all requirements installed.
    - Run the server code to start object detection/counting (.py)

## Usage:

    Access the ESP32-CAM images through HTTP endpoints provided by the server.
    Real-time object counts will be displayed on the server's graphical user interface.

Feel free to customize the project for different objects or resolutions based on your specific requirements.
