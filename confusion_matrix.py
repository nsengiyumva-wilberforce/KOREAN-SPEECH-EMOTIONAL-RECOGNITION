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

# Normalize the data by row (true emotions) so each row sums to 1
row_sums = data.sum(axis=1)[:, np.newaxis]
normalized_data = data / row_sums

# Set up the plot
plt.figure(figsize=(10, 7))

# Create a brick-colored colormap
brick_cmap = sns.light_palette("firebrick", as_cmap=True)

# Generate the heatmap using the normalized data
# vmin=0 and vmax=1 keep the color scale locked from 0% to 100%
sns.heatmap(normalized_data, 
            annot=True, 
            fmt=".1%",  # Automatically formats the decimals as percentages (e.g., 41.1%)
            cmap=brick_cmap, 
            vmin=0, 
            vmax=1,
            xticklabels=classes, 
            yticklabels=classes,
            cbar_kws={'label': 'Percentage of True Emotion'})

# Formatting labels and title
plt.title('Normalized Emotion Confusion Matrix', fontsize=14, pad=15)
plt.xlabel('Predicted Emotion', fontsize=12, labelpad=10)
plt.ylabel('True Emotion', fontsize=12, labelpad=10)

# Adjust ticks to be more readable
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()

# Save the plot as a PNG image
output_filename = 'normalized_brick_confusion_matrix.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Normalized confusion matrix successfully saved as {output_filename}")