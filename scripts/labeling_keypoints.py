# Definerer keypoints med deres pixel koordinater
keypoints = {
    1: (44, 30), 2: (457, 30), 3: (44, 563), 4: (457, 563), 5: (44, 138), 6: (174, 138), 7: (174, 230),
    8: (174, 346), 9: (174, 453), 10: (44, 453), 11: (44, 230), 12: (86, 230), 13: (86, 368),
    14: (44, 368), 15: (130, 295), 16: (457, 230), 17: (457, 295), 18: (457, 368), 19: (201, 295),
    20: (387, 295), 21: (530, 295), 22: (870, 30), 23: (870, 563), 24: (870, 138), 25: (740, 138),
    26: (740, 230), 27: (740, 346), 28: (740, 453), 29: (870, 453), 30: (870, 230), 31: (827, 230),
    32: (827, 368), 33: (870, 368), 34: (785, 295), 35: (712, 295), 36: (601, 121)
}

# Bilde dimensjoner
img_width, img_height = 1920, 1080

# Leser en eksempel-label
label_lines = [
    "17 0.472477 0.476273 0.023359 0.033565",
    "20 0.659091 0.367241 0.019307 0.078944",
    "15 0.471411 0.270051 0.023573 0.029083",
    "19 0.288427 0.363810 0.017073 0.080028",
    "1 0.469932 0.129769 0.016250 0.020833"
]

# Behandler hver linje i labelen for å legge til keypoints
updated_labels = []
for line in label_lines:
    parts = line.split()
    class_index = parts[0]
    new_line = line
    # Legger til alle keypoints som relative koordinater
    for kp_index in range(1, 37):
        if kp_index in keypoints:
            kp_x, kp_y = keypoints[kp_index]
            # Konverterer til relative koordinater
            rel_x = kp_x / img_width
            rel_y = kp_y / img_height
            new_line += f" {rel_x:.6f} {rel_y:.6f}"
    updated_labels.append(new_line)

# Skriver ut de oppdaterte label-linjene
for label in updated_labels:
    print(label)
