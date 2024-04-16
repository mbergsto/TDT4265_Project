keypoints = {
    1: (45, 30), 2: (458, 29), 3: (43, 565), 4: (456, 561), 5: (44, 138),
    6: (174, 137), 7: (175, 235), 8: (175, 347), 9: (175, 454), 10: (45, 452),
    11: (46, 225), 12: (84, 226), 13: (88, 368), 14: (45, 367), 15: (130, 295),
    16: (458, 227), 17: (457, 294), 18: (457, 367), 19: (201, 293), 20: (387, 297),
    21: (530, 295), 22: (870, 29), 23: (871, 565), 24: (870, 138), 25: (741, 139),
    26: (740, 237), 27: (739, 346), 28: (742, 452), 29: (870, 454), 30: (870, 225),
    31: (828, 225), 32: (826, 367), 33: (870, 368), 34: (785, 296), 35: (712, 296),
    36: (601, 121)
}

def adjust_coordinates(kp_dict, tolerance=8):
    from collections import defaultdict

    # Funksjon for å finne grupper av lignende koordinater
    def group_coordinates(values):
        sorted_values = sorted(set(values))
        grouped = defaultdict(list)
        group_key = 0
        grouped[group_key].append(sorted_values[0])

        for i in range(1, len(sorted_values)):
            if sorted_values[i] - sorted_values[i-1] <= tolerance:
                grouped[group_key].append(sorted_values[i])
            else:
                group_key += 1
                grouped[group_key].append(sorted_values[i])

        # Finn gjennomsnittet av hver gruppe for å standardisere
        average_group = {key: round(sum(group)/len(group)) for key, group in grouped.items()}
        new_values = {}
        for key, group in grouped.items():
            for val in group:
                new_values[val] = average_group[key]
        return new_values

    x_coords = [pos[0] for pos in kp_dict.values()]
    y_coords = [pos[1] for pos in kp_dict.values()]

    # Få justerte koordinater
    adjusted_x = group_coordinates(x_coords)
    adjusted_y = group_coordinates(y_coords)

    # Oppdater keypoints med justerte koordinater
    adjusted_dict = {id: (adjusted_x[pos[0]], adjusted_y[pos[1]]) for id, pos in kp_dict.items()}
    return adjusted_dict

# Juster keypoints og opprett dictionary
adjusted_keypoints = adjust_coordinates(keypoints)
print(adjusted_keypoints)
