import cv2 


image = cv2.imread("objects.png")

cv2.imshow("image",image)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.imshow("gray",gray)

_,thresh=cv2.threshold(gray,230,250,cv2.THRESH_BINARY_INV)

cv2.imshow("Thresh",thresh)

contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


for c in contours:
    if cv2.contourArea(c)<200:
        break
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.02*peri,True)
    vertices = len(approx)

    if vertices == 3:
        shape = 'Triangle'
    elif vertices == 4:
        shape = 'Rectangle'
    elif vertices == 5:
        shape = 'Pentagon'
    elif vertices == 6:
        shape ='Hexagon'
    else:
        shape ='Circle'


    M = cv2.moments(c)
    cx= int(M["m10"]/M["m00"])
    cy= int(M["m01"]/M["m00"])

    cv2.putText(image,shape,(cx-40,cy),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)


cv2.drawContours(image,contours,-1,(0,255,0),3)
cv2.imshow("shapes_detected",image)


cv2.waitKey(0)
while True:
    key=cv2.waitKey(1) & 0xFF

    if key == 27:
        break
cv2.destroyAllWindows()
