import cv2
import numpy as np


img = cv2.imread("Shapes.jpg")

img = cv2.resize(img, (400, 400))


# Convert to grayscale
gray_img = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

# Remove noise
gray_img = cv2.medianBlur(
    gray_img,
    5
)


# Threshold
_, binary = cv2.threshold(
    gray_img,
    127,
    255,
    cv2.THRESH_BINARY_INV
)


# Morphological Closing
kernel = np.ones(
    (3, 3),
    np.uint8
)

binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel
)


# Find contours
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


triangle_c = 0
square_c = 0
rect_c = 0
circ_c = 0


for c in contours:

    area = cv2.contourArea(c)

    if area < 200 or area > 3000:
        continue


    # Draw contour
    cv2.drawContours(
        img,
        [c],
        -1,
        (0, 255, 0),
        2
    )


    # Perimeter
    per = cv2.arcLength(
        c,
        True
    )


    # Approximate contour
    approx = cv2.approxPolyDP(
        c,
        0.04 * per,
        True
    )
    print("Corners:", len(approx))


    # Position for text
    x, y, w, h = cv2.boundingRect(c)


    # =========================
    # Triangle
    # =========================

    if len(approx) == 3:

        triangle_c += 1
        shape = "Triangle"


    # =========================
    # Square / Rectangle
    # =========================

    elif len(approx) == 4:

        rect = cv2.minAreaRect(c)

        rw, rh = rect[1]

        ratio = max(rw, rh) / min(rw, rh)

        if ratio < 2.5:
            square_c += 1
            shape = "Square"
        else:
            rect_c += 1
            shape = "Rectangle"

    # =========================
    # Circle
    # =========================

    else:

        if per != 0:

            circularity = (
                4 * np.pi * area
                / (per ** 2)
            )

        else:

            circularity = 0


        if circularity > 0.75:

            circ_c += 1
            shape = "Circle"

        else:

            shape = "Unknown"


    # Write shape name
    cv2.putText(
        img,
        shape,
        (x, y + h + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 0, 0),
        1
    )


# =========================
# Counts
# =========================

cv2.putText(
    img,
    f"Triangles: {triangle_c}",
    (10, 20),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 0, 0),
    1
)

cv2.putText(
    img,
    f"Squares: {square_c}",
    (10, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 0, 0),
    1
)

cv2.putText(
    img,
    f"Rectangles: {rect_c}",
    (10, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 0, 0),
    1
)

cv2.putText(
    img,
    f"Circles: {circ_c}",
    (10, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 0, 0),
    1
)


# =========================
# Show
# =========================

cv2.imshow("Binary", binary)
cv2.imshow("Detected Shapes", img)


# =========================
# Save
# =========================

cv2.imwrite(
    "Detected Shapes.jpg",
    img
)

cv2.imwrite(
    "Binary Shapes.jpg",
    binary
)


cv2.waitKey(0)
cv2.destroyAllWindows()
