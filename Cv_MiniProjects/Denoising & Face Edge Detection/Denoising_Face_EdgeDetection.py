import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load noisy image
image = cv2.imread("noisy_messi.jpg")

# Resize
image = cv2.resize(image, (400, 600))




average_image = cv2.blur(image, (5, 5))


median_image = cv2.medianBlur(image, 5)


gaussian_image = cv2.GaussianBlur(image, (5, 5), 0)




average_gray = cv2.cvtColor(
    average_image,
    cv2.COLOR_BGR2GRAY
)

median_gray = cv2.cvtColor(
    median_image,
    cv2.COLOR_BGR2GRAY
)

gaussian_gray = cv2.cvtColor(
    gaussian_image,
    cv2.COLOR_BGR2GRAY
)




# Average
edge_average = cv2.Canny(
    average_gray,
    70,
    150
)

# Median
edge_median = cv2.Canny(
    median_gray,
    60,
    130
)

# Gaussian
edge_gaussian = cv2.Canny(
    gaussian_gray,
    60,
    130
)




plt.figure(figsize=(12, 8))


# Original
plt.subplot(2, 4, 1)
plt.imshow(
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
)
plt.title("Original")
plt.axis("off")


# Average
plt.subplot(2, 4, 2)
plt.imshow(
    average_gray,
    cmap="gray"
)
plt.title("Average")
plt.axis("off")


# Median
plt.subplot(2, 4, 3)
plt.imshow(
    median_gray,
    cmap="gray"
)
plt.title("Median")
plt.axis("off")


# Gaussian
plt.subplot(2, 4, 4)
plt.imshow(
    gaussian_gray,
    cmap="gray"
)
plt.title("Gaussian")
plt.axis("off")


# Canny Average
plt.subplot(2, 4, 5)
plt.imshow(
    edge_average,
    cmap="gray"
)
plt.title("Canny Average")
plt.axis("off")


# Canny Median
plt.subplot(2, 4, 6)
plt.imshow(
    edge_median,
    cmap="gray"
)
plt.title("Canny Median")
plt.axis("off")


# Canny Gaussian
plt.subplot(2, 4, 7)
plt.imshow(
    edge_gaussian,
    cmap="gray"
)
plt.title("Canny Gaussian")
plt.axis("off")

plt.savefig(
    "messi_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.tight_layout()
plt.show()