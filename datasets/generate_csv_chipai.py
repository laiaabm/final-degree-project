import os
import pandas as pd

# Directory containing all your tile folders
root_directory = "/ictstr01/groups/labs/marr/qscd01/datasets/241002_hecker_CHIP/stained_chip_smears_sqlite/preprocessed_tiles/"

image_ids = []
image_labels = []
image_paths = []

count = 0

# Walk through the directory structure
for dirpath, dirnames, files in sorted(os.walk(root_directory)):
    for file in files:
        # Full path to the current file
        filepath = os.path.join(dirpath, file)
        # Use the folder name as slide_id
        slide_id = os.path.basename(dirpath)
        
        # Determine label based on whether slide id starts with '!'
        if slide_id.startswith("!"):
            label = "chipnegative"
        else:
            label = "chippositive"

        # Append the values to the lists
        image_ids.append(slide_id)
        image_labels.append(label)
        image_paths.append(filepath)

        count += 1

print("Total tile count:", count)

# Prepare your data dictionary
data = {
    'slide_id': image_ids, 
    'label': image_labels, 
    'tile_path': image_paths
}

# Create a DataFrame
df = pd.DataFrame(data)

# Write out the CSV in the parent directory
csv_path = "/ictstr01/groups/labs/marr/qscd01/datasets/241002_hecker_CHIP/stained_chip_smears_sqlite/data_tiles.csv"
df.to_csv(csv_path, index=False)

print(f"Data has been written to {csv_path}")
