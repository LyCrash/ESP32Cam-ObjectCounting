from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display
import cv2
import requests
import numpy as np
from io import BytesIO

# Replace with the actual ESP32-CAM IP address and port
esp32_cam_url = "http://192.168.1.36/cam-hi.jpg"

# Init your workflow
wf = Workflow()

# Add object detection algorithm
detector = wf.add_task(name="infer_yolo_v7", auto_connect=True)

# Add ByteTrack tracking algorithm
tracking = wf.add_task(name="infer_deepsort", auto_connect=True)

tracking.set_parameters({
    "categories": "car, truck, bus",
    "conf_thres": "0.2",
})

# Initialize counts for each category
car_count = 0
truck_count = 0
bus_count = 0

while True:
    # Fetch image from ESP32-CAM via HTTP request
    response = requests.get(esp32_cam_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Convert image data to numpy array
        img_array = np.frombuffer(response.content, dtype=np.uint8)

        # Decode the image
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Run the workflow on the current frame
        wf.run_on(array=frame)

        # Get results
        image_out = tracking.get_output(0)
        obj_detect_out = tracking.get_output(1)

        # Convert the result to BGR color space for displaying
        img_out = image_out.get_image_with_graphics(obj_detect_out)
        img_res = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

        # Get detected objects
        detected_objects = obj_detect_out.get_objects()

        # Loop through detected objects and update counts
        for obj in detected_objects:
            label = obj.label
            if label == "car":
                car_count = max(car_count, obj.id)
            elif label == "truck":
                truck_count = max(truck_count, obj.id)
            elif label == "bus":
                bus_count = max(bus_count, obj.id)

        # Display counts
        counts_text = f"Car: {car_count}, Truck: {truck_count}, Bus: {bus_count}"
        cv2.putText(img_res, counts_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display
        display(img_res, title="DeepSORT", viewer="opencv")

        # Press 'q' to quit the image processing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print(
            f"Error: Unable to fetch image. Status code: {response.status_code}")

# Release resources
cv2.destroyAllWindows()
