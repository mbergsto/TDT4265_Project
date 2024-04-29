import os
import random
import time 
from ultralytics import YOLO
import time


def calculate_inference():
    # Load the model
    model = YOLO("yolov8n.pt")

    # Start timer
    start_time = time.time()

    # Perform inference on random image
    random_image_path = random.choice(os.listdir("data_yolov8/object_datasets/1_train-val_1min_aalesund_from_start/images/all_frames"))
    print("Performing inference on image: {}".format(random_image_path))
    outputs = model.predict("data_yolov8/object_datasets/1_train-val_1min_aalesund_from_start/images/all_frames/"+random_image_path)

    # End timer
    end_time = time.time()
    print("Inference time: {:.2f} seconds".format(end_time - start_time))

calculate_inference()
