import os

text_file_path = "/datasets/tdt4265/other/rbk/3_test_1min_hamkam_from_start/gt/gt.txt"  
img_width = 1920  
img_height = 1080  

def generate_txt_files_for_train_val(text_file_path, img_width, img_height, base_path, train_count=1400):
    """
    Konverterer bildeannotasjoner fra en spesifisert tekstfil til YOLO-formaterte .txt-filer for trening og validering.

    Parametere:
    - text_file_path (str): Sti til tekstfilen som inneholder annotasjoner.
    - img_width (int): Bredde på bildene som annotasjonene refererer til.
    - img_height (int): Høyde på bildene som annotasjonene refererer til.
    - base_path (str): Base sti hvor de genererte .txt-filene for trening og validering vil bli lagret.
    - train_count (int): Antall frames som skal brukes til trening. Resten vil bli brukt til validering.

    Tekstfilformat:
    - Forventer linjer med format: frame_number, object_id, x, y, width, height, active_status, class_id, additional_info.
    - `x, y` er øvre venstre hjørne av bounding boxen, og `width, height` er boksens dimensjoner.
    - `class_id` justeres fra 1-basert til 0-basert indeksering for YOLO-kompatibilitet.

    Utdata:
    - Genererer .txt-filer i YOLO-format (class_id x_center y_center width height) for både trening og validering. 
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
    train_frames = sorted_frame_numbers[:train_count]
    val_frames = sorted_frame_numbers[train_count:]
    
    # Lagre .txt for trening
    for frame_number in sorted_frame_numbers:
        formatted_frame_number = str(frame_number).zfill(6)
        output_file_path = os.path.join(base_path, "labels", "test", f"{formatted_frame_number}.txt")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w') as output_file:
            output_file.write("\n".join(annotations_by_frame[frame_number]))

    # Lagre .txt for validering
    # for frame_number in val_frames:
    #     formatted_frame_number = str(frame_number).zfill(6)
    #     output_file_path = os.path.join(base_path, "labels", "val", f"{formatted_frame_number}.txt")
    #     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    #     with open(output_file_path, 'w') as output_file:
    #         output_file.write("\n".join(annotations_by_frame[frame_number]))

# Kall funksjonen med oppdaterte stier og verdier
base_path = "data_yolov8/3_test_1min_hamkam_from_start/"  
generate_txt_files_for_train_val(text_file_path, img_width, img_height, base_path)
