import cv2

cap = cv2.VideoCapture("AUR_vid.mp4")

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"x = {x}, y = {y}")

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", 800, 600)
cv2.setMouseCallback("Video", mouse_callback)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Video", frame)

    key = cv2.waitKey(30) & 0xFF

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()