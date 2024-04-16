import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Define the keypoints
keypoints = {
    1: (44, 30), 2: (457, 30), 3: (44, 563), 4: (457, 563),
    5: (44, 138), 6: (174, 138), 7: (174, 230), 8: (174, 346),
    9: (174, 453), 10: (44, 453), 11: (44, 230), 12: (86, 230),
    13: (86, 368), 14: (44, 368), 15: (130, 295), 16: (457, 230),
    17: (457, 295), 18: (457, 368), 19: (201, 295), 20: (387, 295),
    21: (530, 295), 22: (870, 30), 23: (870, 563), 24: (870, 138),
    25: (740, 138), 26: (740, 230), 27: (740, 346), 28: (740, 453),
    29: (870, 453), 30: (870, 230), 31: (827, 230), 32: (827, 368),
    33: (870, 368), 34: (785, 295), 35: (712, 295)
}

# Create a figure and axis
fig, ax = plt.subplots()

ax.add_patch(plt.Rectangle((44, 30), 826, 533, fill=False))

# # Plot the 16 meter box
# ax.add_patch(plt.Rectangle((44, 138), 130, 315, fill=False))
# ax.add_patch(plt.Rectangle((870, 138), 130, 315, fill=False))

# # # Plot the center circle
# # center_circle = Circle((457, 295), 73, fill=False)
# # ax.add_patch(center_circle)

# # Plot 5 meter boxes
# ax.add_patch(plt.Rectangle((44, 230), 42, 138, fill=False))
# ax.add_patch(plt.Rectangle((870, 230), 40, 138, fill=False))



plt.show()