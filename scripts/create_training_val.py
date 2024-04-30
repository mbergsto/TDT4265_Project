import os
import random
import shutil
import cv2

def generate_shuffled_txt_img_files_for_train_val(text_file_paths, img_width, img_height, base_path, all_images_paths, players_only=False, ball_only=False, val_ratio=0.2):
    annotations_by_frame = {}
    dataset_prefix = 1

    for text_file_path, all_images_path in zip(text_file_paths, all_images_paths):
        with open(text_file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                frame_number, object_id, x, y, width, height, active_status, class_id, additional_info = parts
                class_id = int(class_id) - 1
                if dataset_prefix == 2:
                    frame_number = int(frame_number) + 1802

                x_center = (float(x) + float(width) / 2) / img_width
                y_center = (float(y) + float(height) / 2) / img_height
                width_norm = float(width) / img_width
                height_norm = float(height) / img_height

                if players_only:
                    if class_id == 1:
                        class_id = 0
                        yolo_format = f"{class_id} {x_center} {y_center} {width_norm} {height_norm}"
                        annotations_by_frame[frame_number] = [yolo_format]
                
                elif ball_only:
                    if class_id == 0:
                        yolo_format = f"{class_id} {x_center} {y_center} {width_norm} {height_norm}"
                        annotations_by_frame[frame_number] = [yolo_format]
                else:
                    yolo_format = f"{class_id} {x_center} {y_center} {width_norm} {height_norm}"
                    annotations_by_frame[frame_number] = [yolo_format]
            
        dataset_prefix += 1
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
    ensure_all_frames_directory(images_path, all_images_paths)

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

def ensure_all_frames_directory(all_frames_path, all_images_paths):
    dateset_prefix = 0
    for all_images_path in all_images_paths:  
        dateset_prefix += 1
        for image_name in os.listdir(all_images_path):
            src_image_path = os.path.join(all_images_path, image_name)
            if dateset_prefix == 2:
                frame_number = int(image_name.split('.')[0]) + 1802
                formatted_frame_number = str(frame_number).zfill(6)
                image_name = f"{formatted_frame_number}.jpg"
            dst_image_path = os.path.join(all_frames_path, image_name)
            shutil.copy2(src_image_path, dst_image_path)
    




txt_file_paths = ["/datasets/tdt4265/other/rbk/1_train-val_1min_aalesund_from_start/gt/gt.txt", "/datasets/tdt4265/other/rbk/2_train-val_1min_after_goal/gt/gt.txt"]
img_folder_paths = ["/datasets/tdt4265/other/rbk/1_train-val_1min_aalesund_from_start/img1/", "/datasets/tdt4265/other/rbk/2_train-val_1min_after_goal/img1/"]
object_path = f"data/object_dataset/"  
ball_path = f"data/ball_dataset/"  
player_path = f"data/player_dataset/"


img_width = 1920  
img_height = 1080  

generate_shuffled_txt_img_files_for_train_val(txt_file_paths, img_width, img_height, object_path, img_folder_paths)
generate_shuffled_txt_img_files_for_train_val(txt_file_paths, img_width, img_height, ball_path, img_folder_paths, ball_only=True)
generate_shuffled_txt_img_files_for_train_val(txt_file_paths, img_width, img_height, player_path, img_folder_paths, players_only=True)

