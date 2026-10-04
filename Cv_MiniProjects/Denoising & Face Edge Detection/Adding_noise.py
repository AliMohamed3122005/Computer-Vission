import cv2
import numpy as np

image = cv2.imread("Messi.jfif")

noise = np.random.normal(0,25,image.shape)

noisy_image = image.astype(np.float32) + noise

noisy_image = np.clip(noisy_image, 0, 255)
noisy_image = noisy_image.astype(np.uint8)

cv2.imshow("messi",noisy_image)

cv2.imwrite("noisy_messi.jpg",noisy_image)

cv2.waitKey(0)
cv2.destroyAllWindows()