import cv2
import numpy as np


image = cv2.imread("images.jfif")


def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("BGR:", image[y, x])
        print("HSV:", image_hsv[y, x])

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse)
cv2.imshow("image", image)


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

lower_rgb = np.array([0, 0, 200])
upper_rgb = np.array([100, 255, 255])
mask_rgb = cv2.inRange(image_rgb, lower_rgb, upper_rgb)
marked_rgb=cv2.bitwise_and(image_rgb,image_rgb,mask=mask_rgb)
cv2.imshow("mask_rgb",marked_rgb)


image_hsv= cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
cv2.imshow("hsv",image_hsv)



lower_hsv = np.array([90, 200, 70])
upper_hsv = np.array([100, 255, 150])


mask_hsv= cv2.inRange(image_hsv, lower_hsv, upper_hsv)
marked_hsv= cv2.bitwise_and(image, image, mask=mask_hsv)

cv2.imshow("mask_hsv",marked_hsv)




cv2.waitKey(0)

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

cv2.destroyAllWindows()