import os
import shutil

def generate_txt_img_files_for_test(text_file_path, img_width, img_height, base_path, all_images_path):
    """
    Konverterer bildeannotasjoner fra en spesifisert tekstfil til YOLO-formaterte .txt-filer for trening og validering.

    Parametere:
    - text_file_path (str): Sti til tekstfilen som inneholder annotasjoner.
    - img_width (int): Bredde på bildene som annotasjonene refererer til.
    - img_height (int): Høyde på bildene som annotasjonene refererer til.
    - base_path (str): Base sti hvor de genererte .txt-filene for trening og validering vil bli lagret.

    Tekstfilformat:
    - Forventer linjer med format: frame_number, object_id, x, y, width, height, active_status, class_id, additional_info.
    - `x, y` er øvre venstre hjørne av bounding boxen, og `width, height` er boksens dimensjoner.
    - `class_id` justeres fra 1-basert til 0-basert indeksering for YOLO-kompatibilitet.

    Utdata:
    - Genererer .txt-filer i YOLO-format (class_id x_center y_center width height) for hvert bilde.
      Det genereres en fil pr. bilde og hver linje representerer en annotasjon, normalisert
      til [0, 1] basert på bildedimensjonene.
    """

    annotations_by_frame = {}
    
    with open(text_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            frame_number, object_id, x, y, width, height, active_status, class_id, additional_info = parts
            class_id = int(class_id) - 1  # YOLO forventer at klassene starter fra 0

            x_center = (float(x) + (float(width) / 2)) / img_width
            y_center = (float(y) + (float(height) / 2)) / img_height
            width_norm = float(width) / img_width
            height_norm = float(height) / img_height

            yolo_format = f"{class_id} {x_center} {y_center} {width_norm} {height_norm}"

            if frame_number not in annotations_by_frame:
                annotations_by_frame[frame_number] = []
            annotations_by_frame[frame_number].append(yolo_format)

    sorted_frame_numbers = sorted(annotations_by_frame.keys(), key=lambda x: int(x))
    
    # Lagre .txt for testing
    for frame_number in sorted_frame_numbers:
        formatted_frame_number = str(frame_number).zfill(6)
        output_file_path = os.path.join(base_path, "labels", "test", f"{formatted_frame_number}.txt")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w') as output_file:
            output_file.write("\n".join(annotations_by_frame[frame_number]))
    
    # Distribuer bildene på samme måte som annotasjonene
    test_images_path = os.path.join(base_path, "images", "test")
    os.makedirs(test_images_path, exist_ok=True)
    ensure_all_frames_directory(test_images_path, all_images_path)

def ensure_all_frames_directory(test_images_path, all_images_path):
    if not os.listdir(test_images_path):
        # all_frames mappen er tom, kopier alle bilder fra all_images_path til test_images_path
        for image_name in os.listdir(all_images_path):
            src_image_path = os.path.join(all_images_path, image_name)
            dst_image_path = os.path.join(test_images_path, image_name)
            shutil.copy2(src_image_path, dst_image_path)



# Kall funksjonen med oppdaterte stier og verdier
base_path = "data_yolov8/3_test_1min_hamkam_from_start/"  
text_file_path = "/datasets/tdt4265/other/rbk/3_test_1min_hamkam_from_start/gt/gt.txt"
all_images_path = "/datasets/tdt4265/other/rbk/3_test_1min_hamkam_from_start/img1/"
img_width = 1920  
img_height = 1080  
generate_txt_img_files_for_test(text_file_path, img_width, img_height, base_path, all_images_path)
