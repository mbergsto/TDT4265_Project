# Import necessary libraries

from ultralytics import YOLO
import os
import random
import numpy as np
import shutil
import supervision as sv
import cv2
import matplotlib.pyplot as plt

# Import function to generate shuffled text and image files for training and validation
from scripts.create_training_val import generate_shuffled_txt_img_files_for_train_val

# Define paths for text files and image folders
txt_file_paths = ["/datasets/tdt4265/other/rbk/1_train-val_1min_aalesund_from_start/gt/gt.txt", "/datasets/tdt4265/other/rbk/2_train-val_1min_after_goal/gt/gt.txt"]
img_folder_paths = ["/datasets/tdt4265/other/rbk/1_train-val_1min_aalesund_from_start/img1/", "/datasets/tdt4265/other/rbk/2_train-val_1min_after_goal/img1/"]

# Define paths for object, ball, and player datasets
object_path = f"data_yolov8/object_dataset/"  
ball_path = f"data_yolov8/ball_dataset/"  
player_path = f"data_yolov8/player_dataset/"

# Define image dimensions
img_width = 1920  
img_height = 1080  

# Generate shuffled text and image files for training and validation for object, ball, and player datasets
generate_shuffled_txt_img_files_for_train_val(txt_file_paths, img_width, img_height, object_path, img_folder_paths)
generate_shuffled_txt_img_files_for_train_val(txt_file_paths, img_width, img_height, ball_path, img_folder_paths, ball_only=True)
generate_shuffled_txt_img_files_for_train_val(txt_file_paths, img_width, img_height, player_path, img_folder_paths, players_only=True)

# Define paths for yaml files
yaml_file_combined = "yaml_files/combined.yaml"
yaml_file_ball = "yaml_files/ball.yaml"
yaml_file_player = "yaml_files/player.yaml"

# Define paths for test yaml files
test_combined = "yaml_files/test_files/test_combined.yaml"
test_ball = "yaml_files/test_files/test_ball.yaml"
test_players = "yaml_files/test_files/test_players.yaml"

# Import function to generate text and image files for testing
from scripts.test_dataset import generate_txt_img_files_for_test

# Define paths for object, player and ball testing datasets
object_path = f"data_yolov8/object_dataset/3_test_1min_hamkam_from_start/"  
ball_path = f"data_yolov8/ball_dataset/3_test_1min_hamkam_from_start/"  
player_path = f"data_yolov8/player_dataset/3_test_1min_hamkam_from_start/" 
text_file_path = "/datasets/tdt4265/other/rbk/3_test_1min_hamkam_from_start/gt/gt.txt"  
all_images_path = "/datasets/tdt4265/other/rbk/3_test_1min_hamkam_from_start/img1/"

# Generate text and image folders for testing all dataset types
generate_txt_img_files_for_test(text_file_path, img_width, img_height, object_path, all_images_path)
generate_txt_img_files_for_test(text_file_path, img_width, img_height, ball_path, all_images_path, type='ball')
generate_txt_img_files_for_test(text_file_path, img_width, img_height, player_path, all_images_path, type='player')

# Import pretrained yolov8n model, recommended for training

model_combined = YOLO('yolov8n.pt')
model_ball = YOLO('yolov8n.pt')
model_players = YOLO('yolov8n.pt')

# Train the models on the relevant dataset and save the results to runs/detect/"dataset"
# Ball uses hyperparameters from hyperparams_tuned.yaml
results = model_combined.train(data=yaml_file_combined, epochs=50, batch=14, imgsz=(1920, 1080), project='/work/mbergst/TDT4265_Project/runs/detect/combined')
results = model_ball.train(data=yaml_file_ball, cfg = 'hyperparams_tuned.yaml', epochs=50, batch=14, imgsz=(1920, 1080), project='/work/mbergst/TDT4265_Project/runs/detect/ball')
#results = model_players.train(data=yaml_file_player, epochs=50, batch=14, imgsz=(1920, 1080), project='/work/mbergst/TDT4265_Project/runs/detect/player')

# Test the models on the test dataset and save the results to runs/detect/"dataset"/test
test_combined = model_combined.val(data=test_combined, batch=14, imgsz=(1920, 1080), project = '/work/mbergst/TDT4265_Project/runs/detect/combined/test')
test_ball = model_ball.val(data=test_ball, batch=14, imgsz=(1920, 1080), project = '/work/mbergst/TDT4265_Project/runs/detect/ball/test')
#test_players = model_players.val(data=test_players, batch=14, imgsz=(1920, 1080), project = '/work/mbergst/TDT4265_Project/runs/detect/player/test')

# Function to find the latest model trained with the best weights

def find_latest_model_with_best(base_path, type):
    detect_path = os.path.join(base_path, 'runs/detect', type)
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
    return best_model_paths[0][0]

base_path = '/work/mbergst/TDT4265_Project' 


# Tracking

# Choose dataset 1 or 2:
dataset = 2  # 1, 2 or 3 (3 is test dataset)

if dataset == 1:
    data_folder = "1_train-val_1min_aalesund_from_start/"

elif dataset == 2:
    data_folder = "2_train-val_1min_after_goal/"

else:
    data_folder = "3_test_1min_hamkam_from_start/"

print(f"Dataset: {data_folder}")

# Define paths for frames
all_frames = f'/datasets/tdt4265/other/rbk/{data_folder}/img1'

frame_paths = sorted([os.path.join(all_frames, f) for f in os.listdir(all_frames) if f.endswith('.jpg')])

# Find the latest model trained with the best weights
model_combined = YOLO(find_latest_model_with_best(base_path, 'combined'))
model_ball = YOLO(find_latest_model_with_best(base_path, 'ball'))
model_players = YOLO(find_latest_model_with_best(base_path, 'player'))

for frame_path in frame_paths:
   # Get results from both models for the current frame
    for result1, result2 in zip(model_players.track(frame_path, persist=True, stream=True, line_width=1, tracker = 'botsort.yaml'),
                                model_ball.track(frame_path, persist=True, stream=True, line_width=1, tracker = 'botsort.yaml')):
        # Draw the results on the frame
        annotated_frame1 = result1.plot(font_size=1, line_width=1)
        annotated_frame2 = result2.plot(font_size=1, line_width=1)

        combined_frame = cv2.addWeighted(annotated_frame1, 0.5, annotated_frame2, 0.5, 0)  

        # Show the combined frame
        cv2.imshow('Yolov8n Tracker', combined_frame)
        
        # Break the loop if 'ESC' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 27 is the ASCII value for the ESC key
            break

    if key == 27:  # Check again if 'ESC' is pressed
        break

cv2.destroyAllWindows()  # Close all windows

