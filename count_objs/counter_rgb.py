'''
The idea here is to determine the countours of the detected objects to be able to count them
They are turned into a gray scale to minimise the RGB color noises and smoothen the process
'''



import cv2
import urllib.request
import numpy as np

url = 'http://192.168.1.42/'

# Initialize counters for each object
car_count = 0
truck_count = 0
bus_count = 0

cv2.namedWindow("[live] object counting", cv2.WINDOW_AUTOSIZE)

while True:
    img_resp = urllib.request.urlopen(url + 'cam-hi.jpg')
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur and Canny edge detection
    canny = cv2.Canny(cv2.GaussianBlur(gray, (11, 11), 0), 30, 150, 3)

    # Dilate the edges to close gaps in contours
    dilated = cv2.dilate(canny, (1, 1), iterations=2)

    # Find contours
    (Cnt, _) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Iterate through contours and update counters based on object type
    for contour in Cnt:
        area = cv2.contourArea(contour)

        # The area threshold can be adjusted based on a specific scenario
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)

            # Check if the contour corresponds to a car, truck, or bus
            if w > h and w / h > 1.5:
                car_count += 1
            elif w > h and w / h < 1.5:
                truck_count += 1
            elif h > w:
                bus_count += 1

    # Display the count of objects on the top-left corner of the window
    cv2.putText(img, f'Car Count: {car_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f'Truck Count: {truck_count}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f'Bus Count: {bus_count}', (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw countours on the image
    cv2.drawContours(img, Cnt, -1, (0, 255, 0), 2)
    cv2.imshow("mit contour", canny)
    cv2.imshow("live transmission", img)
    key = cv2.waitKey(5)

    if key == ord('q'):
        break

cv2.destroyAllWindows()
