import cv2 
import numpy as np

def Draw_window(image,i,j,color):
    y0=int(0.1*height)+100*i+20
    y1=int(0.1*height)+100*i+70
    x0=int(width*0.3)+j*75+20
    x1=int(width*0.3)+j*75+50
    image[y0:y1,x0:x1]=color

width = 800
height = 600

image = np.zeros((height,width,3),dtype=np.uint8)

sky= (25, 25, 40) # dark navy night sky
ground = (30, 50, 35) # dark green ground
building= (90, 90, 100) # grey building
window_off = (20, 20, 30) # dark window
window_on= (255, 220, 120) # warm yellow light

image[:] = sky 
image[int(0.8*height):,]=ground
image[int(0.08*height):int(0.95*height),int(0.3*width):int(0.68*width)]=building

windows = []
for i in range(5):
    for j in range(4):
        windows.append((i,j))


while True:
    for(i,j) in windows:
        Draw_window(image,i,j,window_off)

    chosen=np.random.choice(len(windows),size=4)
    for idx in chosen:
        i,j=windows[idx]
        Draw_window(image,i,j,window_on)

    dis=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    cv2.imshow("Night",dis)

    if cv2.waitKey(1000) == 27:
        break
cv2.destroyAllWindows()