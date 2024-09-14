import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import pandas as pd

# Pixel counts for the confusion matrix with two classes: background (class 0) and disease (class 1)
# deeplabv3+ LOSS
# conf_matrix_data = np.array([[50515158, 3330438],[3185727, 4834661]])

# maskrcnn LOSS
# conf_matrix_data = np.array([[23612563, 1060590],[2653289, 2008718]])

# deeplabv3+ IOU
conf_matrix_data = np.array([[46120468, 7725128],[2286312, 5734076]])

# maskrcnn IOU
#conf_matrix_data = np.array([[15665124, 4855341],[1281336, 3191087]])

# Calculate percentages based on the total pixel counts in each class
conf_matrix_percentages = conf_matrix_data / conf_matrix_data.sum(axis=1, keepdims=True) * 100

# Create a new matrix for percentage values with counts in brackets
annot_matrix = np.array([[f"TN\n{conf_matrix_data[0, 0]} ({round(conf_matrix_percentages[0, 0], 1)}%)", 
                          f"FP\n{conf_matrix_data[0, 1]} ({round(conf_matrix_percentages[0, 1], 1)}%)"],
                         [f"FN\n{conf_matrix_data[1, 0]} ({round(conf_matrix_percentages[1, 0], 1)}%)", 
                          f"TP\n{conf_matrix_data[1, 1]} ({round(conf_matrix_percentages[1, 1], 1)}%)"]])

# Create the heatmap plot
plt.figure(figsize=(6, 5))
ax = sns.heatmap(conf_matrix_percentages, annot=annot_matrix, fmt="", cmap="Blues", cbar=True, linewidths=1,
            xticklabels=["Background", "Disease"],
            yticklabels=["Background", "Disease"])

cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# Add labels and title
plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Ground Truth")

# Save the updated confusion matrix as an image
plt.savefig("deeplabv3+_mean_iou_matrix.png")

# Display the plot
plt.show()
