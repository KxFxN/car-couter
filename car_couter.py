import numpy as np
import cv2
import cvzone
import math
import time
from yolov8 import ONNX,FPS

# Function to calculate centroid of a bounding box
def get_centroid(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy

def is_above_line(x, y, line):
    x1, y1, x2, y2 = line
    return (line == line1 and x1 <= x <= x2 and y1 - 10 < y <=  y1 + 8) or (line == line2 and x1 <= x <= x2 and y1 + 10 > y >= y1 - 8)

model = ONNX('./Model/yolov8m.onnx',0.5,0.5)

cap = cv2.VideoCapture('./Video/Car_720.mp4')

fps_calculator = FPS() 

line1 = [650, 550, 1200, 550]
line2 = [0,550,550,550]

car_count1 = 0
car_count2 = 0

while True:
    # read Frame
    suc , frame = cap.read()

    # check read frame success
    if not suc:
        break

    # process frame by ONNX runtime
    results = model(frame)

    for result in results:

        box = result.box
        conf = result.conf
        classname = result.cls

        if len(box) > 0 :
            if classname in ["car", "truck", "bus", "motorbike"] and conf > 0.3:

                cx, cy = get_centroid(box)

                cx = int(cx)
                cy = int(cy)

                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                if is_above_line(cx, cy, line1):
                    car_count1 += 1
                elif is_above_line(cx, cy, line2):
                    car_count2 += 1
                
    cv2.putText(frame, f'Car count in: {int(car_count1)}', (line1[0],(line1[1]-30)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    cv2.putText(frame, f'Car count out: {int(car_count2)}', (line2[0],(line2[1]-30)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    cv2.line(frame, (line1[0],line1[1]),(line1[2],line1[3]),(0,255,0),2)
    cv2.line(frame, (line2[0],line2[1]),(line2[2],line2[3]),(0,255,0),2)

    fps = fps_calculator.draw_fps(frame)

    cv2.imshow("Detected Objects", frame)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


