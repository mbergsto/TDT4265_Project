
from ultralytics import YOLO
import os
import random
import cv2
import numpy as np
import shutil
import supervision as sv
import cv2
import matplotlib.pyplot as plt

keypoints = {1: (44, 30), 2: (457, 30), 3: (44, 563), 4: (457, 563), 5: (44, 138), 6: (174, 138), 7: (174, 230), 8: (174, 346), 9: (174, 453), 10: (44, 453), 11: (44, 230), 12: (86, 230), 13: (86, 368), 14: (44, 368), 15: (130, 295), 16: (457, 230), 17: (457, 295), 18: (457, 368), 19: (201, 295), 20: (387, 295), 21: (530, 295), 22: (870, 30), 23: (870, 563), 24: (870, 138), 25: (740, 138), 26: (740, 230), 27: (740, 346), 28: (740, 453), 29: (870, 453), 30: (870, 230), 31: (827, 230), 32: (827, 368), 33: (870, 368), 34: (785, 295), 35: (712, 295)}

# Choose dataset 1 or 2:
dataset = 2  # 1 or 2

if dataset == 1:
    data_folder = "1_train-val_1min_aalesund_from_start"
    yaml_file = "yaml_files/keypoint_1.yaml"
    labels = "labels_1"

else:
    data_folder = "2_train-val_1min_after_goal"
    yaml_file = "yaml_files/keypoint_2.yaml"
    labels = "labels_2"

model = YOLO('yolov8n.pt')

#results = model.train(data=yaml_file, epochs=100, batch=30, imgsz=1280, 
                      #project=f'/work/mbergst/TDT4265_Project/runs/keypoints/{data_folder}')

img_width = 1920  
img_height = 1080  

yaml_file_combined = "yaml_files/combined.yaml"
yaml_file_ball = "yaml_files/ball.yaml"
yaml_file_player = "yaml_files/player.yaml"

test_combined = "yaml_files/test_files/test_combined.yaml"
test_ball = "yaml_files/test_files/test_ball.yaml"
test_players = "yaml_files/test_files/test_players.yaml"

model_combined = YOLO('yolov8n.pt')
model_ball = YOLO('yolov8n.pt')
model_players = YOLO('yolov8n.pt')

# Tren modellen på datasettet
#results = model_combined.train(data=yaml_file_combined, epochs=50, batch=14, imgsz=(1920, 1080), project='/work/mbergst/TDT4265_Project/runs/detect/combined')
#results = model_ball.train(data=yaml_file_ball, epochs=50, batch=14, imgsz=(1920, 1080), project='/work/mbergst/TDT4265_Project/runs/detect/ball')
#results = model_ball.train(data=yaml_file_ball, cfg = 'hyperparams_tuned.yaml', epochs=50, batch=14, imgsz=(1920, 1080), project='/work/mbergst/TDT4265_Project/runs/detect/ball')
#results = model_players.train(data=yaml_file_player, patience=6, epochs=50, batch=14, imgsz=(1920, 1080), project='/work/mbergst/TDT4265_Project/runs/detect/player')


# Test modellen på testsette

# model_ball_old = YOLO("/work/mbergst/TDT4265_Project/runs/detect/ball/train6/weights/best.pt")
# model_ball = YOLO("/work/mbergst/TDT4265_Project/runs/detect/ball/train10/weights/best.pt")
# model_ball_tune = YOLO("/work/mbergst/TDT4265_Project/runs/detect/ball/train12/weights/best.pt")

# test_ball_tune = model_ball_tune.val(data=test_ball, cfg='hyperparameters_tuned.yaml',batch=14, imgsz=(1920, 1080), project = '/work/mbergst/TDT4265_Project/runs/detect/ball/test')
# test_ball_old = model_ball_old.val(data=test_ball, batch=14, imgsz=(1920, 1080), project = '/work/mbergst/TDT4265_Project/runs/detect/ball/test')
# test_ball = model_ball.val(data=test_ball, batch=14, imgsz=(1920, 1080), project = '/work/mbergst/TDT4265_Project/runs/detect/ball/test')


#model_players = YOLO('/work/mbergst/TDT4265_Project/runs/detect/player/train11/weights/best.pt')
#test_players = model_players.val(data=test_players, batch=14, imgsz=(1920, 1080), project = '/work/mbergst/TDT4265_Project/runs/detect/player/test')

#model_combined = YOLO('/work/mbergst/TDT4265_Project/runs/detect/combined/train23/weights/best.pt')
#test_combined = model_combined.val(data=test_combined, batch=14, imgsz=(1920, 1080), project = '/work/mbergst/TDT4265_Project/runs/detect/player/test')


#results = model_ball.train(data=yaml_file_ball, cfg = 'hyperparams_tuned.yaml', epochs=500, patience = 50, batch=14, imgsz=(1920, 1080), project='/work/mbergst/TDT4265_Project/runs/detect/ball')
results = model_combined.train(data=yaml_file_combined, epochs=100, patience=10, batch=14, imgsz=(1920, 1080), project='/work/mbergst/TDT4265_Project/runs/detect/combined')
results = model_players.train(data=yaml_file_player, patience=10, epochs=100, batch=14, imgsz=(1920, 1080), project='/work/mbergst/TDT4265_Project/runs/detect/player')

#model_ball_tune = YOLO("/work/mbergst/TDT4265_Project/runs/detect/ball/train13/weights/best.pt")

#test_ball_tune = model_ball_tune.val(data=test_ball, cfg='hyperparameters_tuned.yaml',batch=14, imgsz=(1920, 1080), project = '/work/mbergst/TDT4265_Project/runs/detect/ball/test')

#model_combined = YOLO("/work/mbergst/TDT4265_Project/runs/detect/combined/train23/weights/best.pt")

#test_combined = model_combined.val(data=test_combined, batch=14, imgsz=(1920, 1080), project = '/work/mbergst/TDT4265_Project/runs/detect/player/test')