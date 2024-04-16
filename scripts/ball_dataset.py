import os
import random
import shutil
import cv2


def generate_ball_dataset(text_file_path, img_width, img_height, base_path, all_images_path, val_ratio=0.2):
    annotations_by_frame = {}
    
    with open(text_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            frame_number, object_id, x, y, width, height, active_status, class_id, additional_info = parts
            class_id = int(class_id) - 1  # Adjusting class id for YOLO (starts from 0)

            x_center = (float(x) + float(width) / 2) / img_width
            y_center = (float(y) + float(height) / 2) / img_height
            width_norm = float(width) / img_width
            height_norm = float(height) / img_height

            yolo_format = f"{class_id} {x_center} {y_center} {width_norm} {height_norm}"

            if class_id == 0:
                if frame_number not in annotations_by_frame:
                    annotations_by_frame[frame_number] = []
                annotations_by_frame[frame_number].append(yolo_format)

    # Shuffle frame numbers to randomly distribute them between train and val sets
    frame_numbers = list(annotations_by_frame.keys())
    random.seed(42)
    random.shuffle(frame_numbers)
    split_index = int(len(frame_numbers) * (1 - val_ratio))
    
    train_frames = frame_numbers[:split_index]
    val_frames = frame_numbers[split_index:]
    
    # Save .txt files for train set
    for frame_number in train_frames:
        formatted_frame_number = str(frame_number).zfill(6)
        output_file_path = os.path.join(base_path, "labels", "train", f"{formatted_frame_number}.txt")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w') as output_file:
            output_file.write("\n".join(annotations_by_frame[frame_number]))

    # Save .txt files for val set
    for frame_number in val_frames:
        formatted_frame_number = str(frame_number).zfill(6)
        output_file_path = os.path.join(base_path, "labels", "val", f"{formatted_frame_number}.txt")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w') as output_file:
            output_file.write("\n".join(annotations_by_frame[frame_number]))
    
    # Distribute the images the same way as the labels

    # If images_path directory does not exist, create it
    images_path = os.path.join(base_path, "images", "all_frames")
    os.makedirs(images_path, exist_ok=True)
    # Ensure all frames are in the all_frames directory, if not copy them from all_images_path
    ensure_all_frames_directory(images_path, all_images_path)

    train_images_path = os.path.join(base_path, "images", "train")
    val_images_path = os.path.join(base_path, "images", "val")
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(val_images_path, exist_ok=True)
    
    for frame_number in train_frames:
        formatted_frame_number = str(frame_number).zfill(6)
        image_path = os.path.join(images_path, f"{formatted_frame_number}.jpg")
        output_image_path = os.path.join(train_images_path, f"{formatted_frame_number}.jpg")
        img = cv2.imread(image_path)
        cv2.imwrite(output_image_path, img)
    
    for frame_number in val_frames:
        formatted_frame_number = str(frame_number).zfill(6)
        image_path = os.path.join(images_path, f"{formatted_frame_number}.jpg")
        output_image_path = os.path.join(val_images_path, f"{formatted_frame_number}.jpg")
        img = cv2.imread(image_path)
        cv2.imwrite(output_image_path, img)

def ensure_all_frames_directory(all_frames_path, all_images_path):
    if not os.listdir(all_frames_path):
        # all_frames mappen er tom, kopier alle bilder fra all_images_path til all_frames_path
        for image_name in os.listdir(all_images_path):
            src_image_path = os.path.join(all_images_path, image_name)
            dst_image_path = os.path.join(all_frames_path, image_name)
            shutil.copy2(src_image_path, dst_image_path)
    

# Kall funksjonen med oppdaterte stier og verdier
base_path = "data_yolov8/ball_datasets/1_train-val_1min_aalesund_from_start/"  
text_file_path = "/datasets/tdt4265/other/rbk/1_train-val_1min_aalesund_from_start/gt/gt.txt"  
all_images_path = "/datasets/tdt4265/other/rbk/1_train-val_1min_aalesund_from_start/img1"
img_width = 1920  
img_height = 1080  
generate_ball_dataset(text_file_path, img_width, img_height, base_path, all_images_path)


            
        
            