import cv2
import numpy as np
import kagglehub
from PIL import Image 
import os, random, shutil
from glob import glob 
from ultralytics import YOLO



# #Data handeling 
# # Paths
# image_dir = "object_tracking/GunData/Images"
# label_dir = "object_tracking/GunData/Labels"
# output_label_dir = "object_tracking/GunData/labels_yolo"

# # Make sure output folder exists
# os.makedirs(output_label_dir, exist_ok=True)

# # Loop through label files
# for filename in os.listdir(label_dir):
#     if not filename.endswith(".txt"):
#         continue

#     label_path = os.path.join(label_dir, filename)
#     image_path = os.path.join(image_dir, filename.replace(".txt", ".jpeg"))
    
#     if not os.path.exists(image_path):
#         print(f"Image not found for {filename}, skipping.")
#         continue

#     # Get image dimensions
#     with Image.open(image_path) as img:
#         width, height = img.size

#     with open(label_path, "r") as f:
#         lines = f.read().strip().split("\n")

#     if len(lines) < 2:
#         continue

#     box_lines = lines[1:]  # skip first line (object count)
#     yolo_lines = []

#     for line in box_lines:
#         x, y, w, h = map(float, line.strip().split())
#         # Convert to YOLO format
#         x_center = (x + w / 2) / width
#         y_center = (y + h / 2) / height
#         w_norm = w / width
#         h_norm = h / height
#         yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

#     # Save to new YOLO label file
#     with open(os.path.join(output_label_dir, filename), "w") as f_out:
#         f_out.write("\n".join(yolo_lines))



# # Paths
# ROOT = os.path.join ("object_tracking","GunData")
# IMG_DIR = os.path.join(ROOT, "images")
# LBL_DIR = os.path.join(ROOT, "labels")   # your YOLO labels folder



# # Output dirs
# TRAIN_IMG = os.path.join(ROOT, "images", "train")
# VAL_IMG   = os.path.join(ROOT, "images", "val")
# TRAIN_LBL = os.path.join(ROOT, "labels", "train")
# VAL_LBL   = os.path.join(ROOT, "labels", "val")
# for d in [TRAIN_IMG,VAL_IMG,TRAIN_LBL,VAL_LBL]:
#     os.makedirs(d, exist_ok=True)

# # Get all .jpeg files
# images = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpeg")]
# images.sort()
# random.shuffle(images)

# # 80/20 split
# split = int(0.8 * len(images))
# train_images = images[:split]
# val_images   = images[split:]

# def copy_pair(img_name, img_dst, lbl_dst):
#     base = os.path.splitext(img_name)[0]
    
#     shutil.copy2(os.path.join(IMG_DIR, img_name), os.path.join(img_dst, img_name))
#     shutil.copy2(os.path.join(LBL_DIR, base + ".txt"), os.path.join(lbl_dst, base + ".txt"))

# for img in train_images:
#     copy_pair(img, TRAIN_IMG, TRAIN_LBL)

# for img in val_images:
#     copy_pair(img, VAL_IMG, VAL_LBL)

# print(f"✅ Done. Train: {len(train_images)}, Val: {len(val_images)}")






model= YOLO("yolov8n.pt")
model.train(data="object_tracking/GunData/data.yaml",
            epochs=30,
           imgsz=512,
            batch=8,
             device="cpu",
              workers=0 )


