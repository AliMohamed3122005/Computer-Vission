from ultralytics import YOLO
import cv2
import csv





model = YOLO("best.pt")
lx1_1=70
lx1_2=230
ly1=90
lx2_1=15
lx2_2=225
ly2=125
video_path = "AUR_Vid.mp4"



class SpeedLine:
    def __init__(self,x1,x2,y):    #line coordinates
        self.x1=x1
        self.x2=x2
        self.y=y


    def is_crossed(self,cx,prev_y,cur_y):
        if self.x1 <= cx <= self.x2:
            if (prev_y < self.y <= cur_y) or (prev_y > self.y >= cur_y) :
                return True
        return False


class Vehicle:
    def __init__(self,vehicle_id):
        self.id = vehicle_id
        self.prev_y = None        #because we don't the position of the car in the previous frame
        self.cross_a_time = None
        self.cross_b_time = None 
        self.speed = None
        self.saved = False
    def calculate_speed(self,distance):
        time = self.cross_b_time - self.cross_a_time
        speed = distance / time * 3.6
        return speed


class SpeedDetector:
    def __init__(self,model_path,video_path):
        self.model = YOLO(model_path)
        self.line_a=SpeedLine(lx1_1,lx1_2,ly1)
        self.line_b=SpeedLine(lx2_1,lx2_2,ly2)
        self.distance = 9.144
        self.frame_count = 0
        self.vehicles = {}
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.file = open("vehicles.csv", "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["ID", "Time A", "Time B", "Speed"])
    def run(self):
        cv2.namedWindow("Vehicle Tracking",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Vehicle Tracking",800,600)
        while True:
            ret,frame = self.cap.read()
            self.frame_count += 1       #for calculating time
            if not ret:
                break
            results = self.model.track(
            frame,
            persist=True,               #saving of ids
            tracker="bytetrack.yaml",
            conf=0.3                    
            )
            boxes = results[0].boxes
            if boxes.id is None:
                continue
            track_ids=boxes.id.int().cpu().tolist()  # Change the tensor type to int(),Tensor may be stay in GPU 
            boxes_xyxy=boxes.xyxy.cpu().tolist()
            vehicle_count=len(track_ids)
            if vehicle_count > 3:
                cv2.putText(
                    frame,
                    "Traffic Jam",
                    (20,25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,0,255),
                    1
                )
            for i in range(len(track_ids)):
                track_id = track_ids[i]
                box = boxes_xyxy[i]
                x1,y1,x2,y2 = box
                cx = (x1+x2) // 2
                cy = (y1+y2) // 2
                if track_id not in self.vehicles:
                    self.vehicles[track_id] = Vehicle(track_id)
                vehicle= self.vehicles[track_id]
                if vehicle.prev_y is not None:                         #because the first frame is None
                    if self.line_a.is_crossed(cx,vehicle.prev_y,cy):   #to check ii the car passed the  line
                        if vehicle.cross_a_time is None:               #to make the time saved just once
                            vehicle.cross_a_time = self.frame_count / self.fps
                    if self.line_b.is_crossed(cx,vehicle.prev_y,cy):
                        if vehicle.cross_b_time is None:
                            vehicle.cross_b_time = self.frame_count / self.fps
                    if vehicle.cross_a_time is not None and vehicle.cross_b_time is not None:
                        speed = vehicle.calculate_speed(self.distance)
                        vehicle.speed = speed
                        if vehicle.speed is not None and not vehicle.saved:
                            self.writer.writerow([
                                vehicle.id,
                                vehicle.cross_a_time,
                                vehicle.cross_b_time,
                                vehicle.speed
                            ])
                            self.file.flush()
                            vehicle.saved = True
                cv2.rectangle(
                        frame,
                        (int(x1),int(y1)),
                        (int(x2),int(y2)),
                        (0,255,0),
                        2
                    )
                label = f"ID:{track_id}"
                if vehicle.speed is not None:
                    label +=f" | Speed: {vehicle.speed:.1f} km/h"
                cv2.putText(
                        frame,
                        label,
                        (int(x1),int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0,0,255),
                        1
                    )
                vehicle.prev_y = cy
            cv2.line(
                        frame,
                        (self.line_a.x1,self.line_a.y),
                        (self.line_a.x2,self.line_a.y),
                        (255,0,0),
                        2
                    )
            cv2.line(
                        frame,
                        (self.line_b.x1,self.line_b.y),
                        (self.line_b.x2,self.line_b.y),
                        (255,0,0),
                        2
                    )
            cv2.imshow("Vehicle Tracking" , frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        self.file.close()
        self.cap.release()
        cv2.destroyAllWindows()


detector = SpeedDetector("best.pt", video_path)
detector.run( )