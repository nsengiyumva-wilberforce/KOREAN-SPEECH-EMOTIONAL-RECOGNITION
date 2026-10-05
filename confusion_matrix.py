import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the classes and the raw counts
classes = ['anger', 'happiness', 'neutral', 'sadness', 'surprise']
data = np.array([
    [30,  9,  22,   4,  8],
    [ 8, 99,  51,  11, 14],
    [63, 110, 388, 115, 44],
    [ 2,  5,  11,  21,  2],
    [12,  7,  29,   7, 25]
])

# Calculate percentages per row (true emotions)
row_sums = data.sum(axis=1)[:, np.newaxis]
percentages = (data / row_sums) * 100

# Create formatted labels combining count and percentage
labels = (np.asarray(["{0}\n{1:.1f}%".format(count, perc) 
                      for count, perc in zip(data.flatten(), percentages.flatten())])
         ).reshape(data.shape)

# Set up the plot
plt.figure(figsize=(10, 7))

# Create a brick-colored colormap
brick_cmap = sns.light_palette("firebrick", as_cmap=True)

# Generate the heatmap
ax = sns.heatmap(data, 
                 annot=labels, 
                 fmt="", 
                 cmap=brick_cmap, 
                 xticklabels=classes, 
                 yticklabels=classes,
                 cbar_kws={'label': 'Number of predictions'})

# Formatting labels and title
plt.title('Emotion Confusion Matrix', fontsize=14, pad=15)
plt.xlabel('Predicted Emotion', fontsize=12, labelpad=10)
plt.ylabel('True Emotion', fontsize=12, labelpad=10)

# Adjust ticks to be more readable
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()

# Save the plot as a PNG image
output_filename = 'brick_confusion_matrix.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Confusion matrix successfully saved as {output_filename}")