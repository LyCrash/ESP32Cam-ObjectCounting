from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display
import cv2, os

# Get the current working directory (project directory)
current_dir = os.getcwd()

# Replace 'your_video_path.mp4' with the actual video file path
input_video_path = os.path.join(os.path.join(
    current_dir, "data"), 'highway.mp4')
output_video_path = os.path.join(os.path.join(
    current_dir, "data"), 'deepsort_highway.avi')

# Init your workflow
wf = Workflow()

# Add object detection algorithm
detector = wf.add_task(name="infer_yolo_v7", auto_connect=True)

# Add ByteTrack tracking algorithm
tracking = wf.add_task(name="infer_deepsort", auto_connect=True)

tracking.set_parameters({
    "categories": "car,truck,bus", # List of object classes
    "conf_thres": "0.7",   # Increase this parameter for more accuracy
})

# Initialize counts for each category
car_count = 0
truck_count = 0
bus_count = 0

# Keep track of the identifier of detected objects (ids)
detected_obj_ids = {
    'car': [],
    'truck': [],
    'bus': []
}
# Set a limit to flush the ids tracking buffer => for performancee issues
threshold = 10
# The buffer pourcentage to flush - from the beginning
drop = threshold//3

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

    # Get the detected objects in the frame
    detected_objects = obj_detect_out.get_objects()


    # Loop through detected objects and update the class counts
    # The same object is not counted simultanisly= tracked by its id
    # The class count is updated once the object class surpasses the trust line
    for obj in detected_objects:
        #print(obj)
        label = obj.label
        ident = obj.id

        if (label == "car") and (ident not in detected_obj_ids["car"]):
            detected_obj_ids["car"].append(ident) # keep track
            car_count += 1 
        elif (label == "truck") and (ident not in detected_obj_ids["truck"]):
            detected_obj_ids["truck"].append(ident)  # keep track
            truck_count += 1
        elif (label == "bus") and (ident not in detected_obj_ids["bus"]):
            detected_obj_ids["bus"].append(ident)  # keep track
            bus_count += 1 

    #print("obj ids: ",detected_obj_ids)

    # Flush the tracking buffer if the threshold is reached
    detected_obj_ids = { k: v[drop:] if len(v)>threshold else v for k,v in detected_obj_ids.items()}
        

    # Display counts
    counts_text = f"Car: {car_count}, Truck: {truck_count}, Bus: {bus_count}"
    cv2.putText(img_res, counts_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the resulting frame
    out.write(img_out)

    # Display
    display(img_res, title="DeepSORT", viewer="opencv")
    #cv2.imshow("DeepSORT", img_res)

    # Press 'q' to quit the video processing
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break


# After the loop release everything
stream.release()
out.release()
cv2.destroyAllWindows()
