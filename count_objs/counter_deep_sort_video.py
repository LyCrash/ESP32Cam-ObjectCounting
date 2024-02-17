from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display
import cv2
import os

# Get the current working directory (project directory)
current_dir = os.getcwd()

# Replace 'your_video_path.mp4' with the actual video file path
input_video_path = os.path.join(os.path.join(
    current_dir, "data"), 'Road_traffic.mp4')
output_video_path = os.path.join(os.path.join(
    current_dir, "data"), 'deepsort_output_Road_traffic.avi')

# Init your workflow
wf = Workflow()

# Add object detection algorithm
detector = wf.add_task(name="infer_yolo_v7", auto_connect=True)

# Add ByteTrack tracking algorithm
tracking = wf.add_task(name="infer_deepsort", auto_connect=True)

tracking.set_parameters({
    "categories": "car, truck, bus",
    "conf_thres": "0.5",
})

# Initialize counts for each category
car_count = 0
truck_count = 0
bus_count = 0

# Open the video file
stream = cv2.VideoCapture(input_video_path)
if not stream.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for the output
frame_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = stream.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc,
                      frame_rate, (frame_width, frame_height))

while True:
    # Read image from stream
    ret, frame = stream.read()

    # Test if the video has ended or there is an error
    if not ret:
        print("Info: End of video or error.")
        break

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

    # Save the resulting frame
    out.write(img_out)

    # Display
    display(img_res, title="DeepSORT", viewer="opencv")

    # Press 'q' to quit the video processing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release everything
stream.release()
out.release()
cv2.destroyAllWindows()
