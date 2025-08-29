import cv2 
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox

#root =tk.Tk()
#root.withdraw()




model= YOLO("runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture(0) 
names=model.names
print(names)

while True:


   ok,frame=cap.read()
   result=model(frame)[0]
   for box in result.boxes:
      class_id =int(box.cls[0])
      class_name=names[class_id]
      ##if class_name=="person":
         #messagebox.showwarning("danger huamn detected!!!!!")
         

   annotated_frame=result.plot()
   frame_resized=cv2.resize(annotated_frame,(1500,800))
   img=cv2.imshow("window",annotated_frame)

   
      


   key=cv2.waitKey(1)

   if key ==27:
      break
   

cap.release()
cv2.destroyAllWindows()
