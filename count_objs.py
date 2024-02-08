import cv2
import urllib.request
import numpy as np

url = 'http://192.168.210.94/'
cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)

while True:
    img_resp = urllib.request.urlopen(url + 'cam-lo.jpg')
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(cv2.GaussianBlur(gray, (11, 11), 0), 30, 150, 3)
    dilated = cv2.dilate(canny, (1, 1), iterations=2)
    (Cnt, _) = cv2.findContours(dilated.copy(),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Display the count of objects on the top-left corner of the window
    cv2.putText(img, f'Objects Count: {len(Cnt)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.drawContours(img, Cnt, -1, (0, 255, 0), 2)

    cv2.imshow("mit contour", canny)
    cv2.imshow("live transmission", img)
    key = cv2.waitKey(5)

    if key == ord('q'):
        break

cv2.destroyAllWindows()
