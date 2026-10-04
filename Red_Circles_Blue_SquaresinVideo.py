import cv2
import numpy as np




class ShapeDetector:
    def __init__(self, video_path):
        self.cap=cv2.VideoCapture(video_path)

    def hsv_frame(self,frame):
        frame = cv2. medianBlur(frame, 5)
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 80, 80])
        upper_red2 = np.array([179, 255, 255])

        red_mask1 = cv2.inRange(
            hsv,
            lower_red1,
            upper_red1
        )

        red_mask2 = cv2.inRange(
            hsv,
            lower_red2,
            upper_red2
        )

        red_mask = red_mask1 | red_mask2

        lower_blue=np.array([90,100,100])
        upper_blue=np.array([130,255,255])
        blue_mask = cv2.inRange(hsv,lower_blue,upper_blue)

        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        blue_mask = cv2.morphologyEx(
            blue_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        blue_mask = cv2.morphologyEx(
            blue_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        red_countour,_=cv2.findContours(red_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        blue_countour,_=cv2.findContours(blue_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        red_countour_filtered= []
        for c in red_countour:
            if cv2.contourArea(c) >100:
                red_countour_filtered.append(c)
        red_countour = red_countour_filtered

        blue_countour_filtered= []
        for c in blue_countour:
            if cv2.contourArea(c) >100:
                blue_countour_filtered.append(c)
        blue_countour = blue_countour_filtered

        for c in red_countour:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c,True)
            if peri == 0:
                continue
            circularity = 4*np.pi*area / (peri**2)
            if circularity >0.85:
                cv2.drawContours(frame,[c],-1,(0,255,0),2)
                x,y,w,h=cv2.boundingRect(c)
                cv2.putText(frame,"Red Circle",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        for c in blue_countour:
            area=cv2.contourArea(c)
            peri = cv2.arcLength(c,True)
            if peri == 0:
                continue
            approx = cv2.approxPolyDP(c,0.04*peri,True)
            if len(approx)==4:
                x,y,w,h=cv2.boundingRect(c)
                ratio = max(w,h)/min(w,h)
                if ratio<=1.2:
                    cv2.drawContours(frame,[c],-1,(0,255,0),2)
                    cv2.putText(frame,"Blue Square",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        return frame

    def run(self):

        while True:
            ret,frame =self.cap.read()
            frame= self.hsv_frame(frame)
            
            if not ret:
                break
                
            cv2.imshow("Shape Detection",frame)
            if cv2.waitKey(30) == 27:
                break
        self.cap.release()
        
        cv2.destroyAllWindows()



if __name__ == "__main__":
    detector = ShapeDetector("video.mp4")
    detector.run()