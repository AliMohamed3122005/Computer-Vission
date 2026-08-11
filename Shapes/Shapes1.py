import cv2
import numpy as np

img = cv2.imread("Shapes.jpg")

img = cv2.resize(img,(400,400))

gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray_img = cv2.medianBlur(gray_img, 5)


_,binary=cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY_INV)

kernel = np.ones((3,3),np.uint8)
binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)  



countors,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

triangle_c = 0
square_c = 0
rect_c = 0
circ_c = 0
for c in countors:

    area = cv2.contourArea(c)
    if area < 200 or area > 3000:
        continue

    cv2.drawContours(img,[c],-1,(0,255,0),2)
    per = cv2.arcLength(c,True)

    approx = cv2.approxPolyDP(c,0.04*per,True)

    x,y,w,h=cv2.boundingRect(c)

    rect=cv2.minAreaRect(c)
    w=rect[1][0]
    h=rect[1][1]
    ratio = float(w)/float(h)

    if len(approx) == 3:
        triangle_c+=1
        shape = "Triangle"

    elif len(approx) == 4:
        ratio = float(w)/float(h)
        if ratio >= 1.1:
            square_c+=1
            shape = "Square"
        else:
            rect_c=rect_c+1
            shape="Rectangle"

    else:
        circularity=4*np.pi * area /(per**2)
        if circularity> 0.75:
            circ_c+=1
            shape="Circle"
        else:
            shape= "unknown"

    cv2.putText(img,shape,(x,y+60),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,0,0),1)



cv2.putText(img,f"Triangles: {triangle_c}",(x+30,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,0,0),1)
cv2.putText(img,f"Squares: {square_c}",(x+30,y-30),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,0,0),1)
cv2.putText(img,f"Rectangles: {rect_c}",(x+30,y+10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,0,0),1)
cv2.putText(img,f"Circles: {circ_c}",(x+30,y+30),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,0,0),1)

cv2.imshow("Binary",binary)
cv2.imshow("Detected Shapes",img)
cv2.imwrite("Detected Shapes.jpg",img)
cv2.imwrite("Binary Shapes.jpg",binary)

cv2.waitKey(0)
