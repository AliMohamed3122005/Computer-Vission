import numpy as np 
import cv2 
import time

height = 800
width = 800
switch = 0  

image = np.zeros((height, width, 3), dtype=np.uint8)



def day(image):
    image[:,:]=(135, 206, 235)  # Sky blue color
    image[height//2:]= (34, 139, 34)  # Green color for ground
    image[height//4:height//2, width//4:3*width//4] = (255, 221, 64)  # Yellow color for sun

def night(image):
    image[:,:]=(35, 29, 43)  # Midnight blue color
    image[height//2:]= (35, 55, 43)  # Black color for ground
    image[height//4:height//2, width//4:3*width//4] = (195, 194, 190)  # White color for moon

while True:
    if switch == 0:
        day(image)
        switch = 1
    else:
        night(image)
        switch = 0
    display_image = image 
    cv2.imshow("Day and Night", display_image)
    if cv2.waitKey(1000)  & 0xFF == 27:
        break
cv2.destroyAllWindows()




