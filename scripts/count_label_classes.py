import os
from collections import Counter, OrderedDict

def count_label_classes(directory_path):
    class_counts = Counter()

    # Går gjennom alle filene i mappen
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            # Åpner hver label-fil
            with open(os.path.join(directory_path, filename), 'r') as file:
                for line in file:
                    class_id = int(line.split()[0])  # Første element i hver linje er class_id og konverterer til int for sortering
                    class_counts[class_id] += 1  # Øker telleverdien for denne class_id

    # Sorter class_counts og returner en OrderedDict
    sorted_class_counts = OrderedDict(sorted(class_counts.items()))
    return sorted_class_counts

# Eksempel på hvordan du bruker funksjonen
directory_path_1 = "/work/mbergst/TDT4265_Project/data_yolov8/keypoint_detection/labels_1"
directory_path_2 = "/work/mbergst/TDT4265_Project/data_yolov8/keypoint_detection/labels_2"

class_counts_1 = count_label_classes(directory_path_1)
print("Labels 1\n", class_counts_1)

class_counts_2 = count_label_classes(directory_path_2)
print("Labels 2\n", class_counts_2)
