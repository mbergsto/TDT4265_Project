import os
import shutil
import random

def distribute_files(image_src_folder, label_src_folder, image_dest_folder, label_dest_folder, ratio=0.8):
    # Definerer stier til trenings- og valideringsmapper for bilder og labels
    train_image_folder = os.path.join(image_dest_folder, 'train')
    val_image_folder = os.path.join(image_dest_folder, 'val')
    train_label_folder = os.path.join(label_dest_folder, 'train')
    val_label_folder = os.path.join(label_dest_folder, 'val')

    # Oppretter mapper hvis de ikke allerede eksisterer
    for folder in [train_image_folder, val_image_folder, train_label_folder, val_label_folder]:
        os.makedirs(folder, exist_ok=True)

    # Henter alle bildefiler og tilhørende txt-filer
    all_images = [file for file in os.listdir(image_src_folder) if file.endswith('.jpg')]
    all_labels = [file for file in os.listdir(label_src_folder) if file.endswith('.txt')]

    # Sikrer at bilde- og txt-filer er sortert likt for korrekt mapping
    all_images.sort()
    all_labels.sort()

    # Blander listene for å sikre tilfeldig fordeling
    combined = list(zip(all_images, all_labels))
    random.seed(42)
    random.shuffle(combined)
    all_images, all_labels = zip(*combined)

    # Beregner indeksen for å splitte datasettet
    split_index = int(len(all_images) * ratio)

    # Fordeler filer til trenings- og valideringsmapper
    for image, label in zip(all_images[:split_index], all_labels[:split_index]):
        shutil.move(os.path.join(image_src_folder, image), os.path.join(train_image_folder, image))
        shutil.move(os.path.join(label_src_folder, label), os.path.join(train_label_folder, label))

    for image, label in zip(all_images[split_index:], all_labels[split_index:]):
        shutil.move(os.path.join(image_src_folder, image), os.path.join(val_image_folder, image))
        shutil.move(os.path.join(label_src_folder, label), os.path.join(val_label_folder, label))

# Kall funksjonen med de faktiske stiene
image_src_folder = '/work/mbergst/TDT4265_Project/data_yolov8/object_datasets/2_train-val_1min_after_goal/images/all_frames'
label_src_folder = '/work/mbergst/TDT4265_Project/data_yolov8/keypoint_detection/1_min_afther_goal_keypoint_annotation'
image_dest_folder = '/work/mbergst/TDT4265_Project/data_yolov8/keypoint_detection/2_train-val_1min_after_goal/images/'
label_dest_folder = '/work/mbergst/TDT4265_Project/data_yolov8/keypoint_detection/2_train-val_1min_after_goal/labels/'
distribute_files(image_src_folder, label_src_folder, image_dest_folder, label_dest_folder)
