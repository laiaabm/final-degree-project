import os
import pandas as pd

# Paths
root_directory = "/ictstr01/home/aih/laia.mana/project/DATA/feature_extraction/features_segmented_40/dinobloom-g/"
mapping_csv = "/ictstr01/home/aih/laia.mana/project/codes/other_codes/file_to_label.csv"
output_csv = "/ictstr01/home/aih/laia.mana/project/DATA/feature_extraction/features_40_dinobloom-g.csv"

# Load filename-to-sample_number mapping
mapping_df = pd.read_csv(mapping_csv)
# Clean up filename (remove .svs extension for easier matching)
mapping_df["patient_id"] = mapping_df["filename"].str.replace(".svs", "", regex=False)

# Create a dictionary for quick lookup: patient_id -> sample_number
sample_number_dict = dict(zip(mapping_df["patient_id"], mapping_df["sample_number"]))

# Initialize list to collect rows
data_list = []

# Walk through files
for dirpath, dirnames, files in sorted(os.walk(root_directory)):
    for file in files:
        if file.endswith(".h5"):
            try:
                slide_id, label_with_ext = file.rsplit("_", 1)
                label = label_with_ext.rsplit(".", 1)[0]
            except ValueError:
                print(f"Unexpected file naming format: {file}")
                continue

            sample_number = sample_number_dict.get(slide_id, "UNKNOWN")  # fallback if not found

            #print("slide_id:", slide_id)
            #print("label:", label)
            #print("sample_number:", sample_number)
            #print("tensor_path:", os.path.join(dirpath, file))

            data_list.append({
                "patient_id": slide_id,
                "label": label,
                "sample_number": sample_number,
                "tensor_paths": os.path.join(dirpath, file)
            })
        else:
            print(f"Skipping non-HDF5 file: {file}")

# Save to CSV
df = pd.DataFrame(data_list)
df.to_csv(output_csv, index=False)
print(f"Data has been written to {output_csv}")
