import cv2
import urllib.request
import numpy as np

url = 'http://192.168.210.94/'
cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)

# Define the classes you want to detect
classes_to_detect = ["car", "truck", "bicycle", "motorbike", "bus"]

# Initialize counts for each class
class_counts = {class_name: 0 for class_name in classes_to_detect}

while True:
    img_resp = urllib.request.urlopen(url + 'cam-lo.jpg')
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(cv2.GaussianBlur(gray, (11, 11), 0), 30, 150, 3)
    dilated = cv2.dilate(canny, (1, 1), iterations=2)
    (Cnt, _) = cv2.findContours(dilated.copy(),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in Cnt:
        # Get bounding box and center coordinates of the contour
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2

        # Your YOLO detection logic here (use YOLO model or any other object detection method)

        # Assuming you have detected the class name for the object
        class_name = "car"  # Replace this with your actual class detection logic

        # If the detected class is in the list of classes to detect
        if class_name in classes_to_detect:
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Increment count for the detected class
            class_counts[class_name] += 1

    # Display counts on the image
    for i, class_name in enumerate(classes_to_detect):
        count_text = f"{class_name}: {class_counts[class_name]}"
        cv2.putText(img, count_text, (10, 30 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("mit contour", canny)
    cv2.imshow("live transmission", img)
    key = cv2.waitKey(5)

    if key == ord('q'):
        break

cv2.destroyAllWindows()
