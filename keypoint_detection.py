# Import necessary libraries
from ultralytics import YOLO
import os
import random
import cv2
import numpy as np
import shutil
import supervision as sv
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time
import pandas as pd
from IPython.display import display

# Choose dataset 1 or 2:
dataset = 1  # 1 or 2

if dataset == 1:
    data_folder = "1_train-val_1min_aalesund_from_start"
    yaml_file = "yaml_files/keypoint_1.yaml"
    labels = "labels_1"

else:
    data_folder = "2_train-val_1min_after_goal"
    yaml_file = "yaml_files/keypoint_2.yaml"
    labels = "labels_2"

# Define paths for image and label folders
image_src_folder = f'/datasets/tdt4265/other/rbk/{data_folder}/img1'
label_src_folder = f'/work/mbergst/TDT4265_Project/data_yolov8/keypoint_labels/{labels}'
image_dest_folder = f'/work/mbergst/TDT4265_Project/data_yolov8/keypoint_detection/{data_folder}/images/'
label_dest_folder = f'/work/mbergst/TDT4265_Project/data_yolov8/keypoint_detection/{data_folder}/labels/'

# Import function to distribute files to image and label folders
from scripts.keypoint_dataset import distribute_files

# Distribute files to image and label folders
distribute_files(image_src_folder, label_src_folder, image_dest_folder, label_dest_folder)

# initialize pretrained model, recommended for training
model = YOLO('yolov8n.pt')

# Train model and save results to runs/keypoints/"dataset", with test_params_keypoints.yaml as configuration file
#results = model.train(data=yaml_file, cfg = 'test_params_keypoints.yaml', epochs=100, batch=30, imgsz=1280, project=f'/work/mbergst/TDT4265_Project/runs/keypoints/{data_folder}')

# Find the latest model with the best performance
def find_latest_model_with_best(base_path, data_folder, type='keypoints'):
    detect_path = os.path.join(base_path, 'runs', type, data_folder)
    training_sessions = [os.path.join(detect_path, d) for d in os.listdir(detect_path) if os.path.isdir(os.path.join(detect_path, d))]
    
    best_model_paths = []

    for session in training_sessions:
        best_model_path = os.path.join(session, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            best_model_paths.append((best_model_path, os.path.getmtime(best_model_path)))

    if not best_model_paths:
        print("Ingen 'best.pt' fil funnet i noen av treningsøktene.")
        return None
    
    best_model_paths.sort(key=lambda x: x[1], reverse=True)
    print(f"Bruker modell fra {best_model_paths[0][0]}")
    return best_model_paths[0][0]


base_path = '/work/mbergst/TDT4265_Project' 


# Tracking

# Get latest model with best weights
model = YOLO(find_latest_model_with_best(base_path, data_folder, type='keypoints'))


# Define path to image folder
all_frames = f'/datasets/tdt4265/other/rbk/{data_folder}/img1'
frame_paths = sorted([os.path.join(all_frames, f) for f in os.listdir(all_frames) if f.endswith('.jpg')])

# Loop through frames and track keypoints
for frame_path in frame_paths:
    for result in model.track(frame_path, persist=True, stream=True, line_width=1):
        annotated_frame = result.plot(font_size=1, line_width=1)
        cv2.imshow('frame', annotated_frame)
       
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  
            break

    if key == 27:
        break

cv2.destroyAllWindows()  





