import os
import pandas as pd


def load_mapping(mapping_csv_path):
    """
    Load and process the mapping file that links filenames to sample numbers.
    """
    mapping_df = pd.read_csv(mapping_csv_path)
    mapping_df["patient_id"] = mapping_df["filename"].str.replace(".svs", "", regex=False)
    return dict(zip(mapping_df["patient_id"], mapping_df["sample_number"]))

def extract_features(root_dir, sample_number_dict):
    """
    Walk through the directory and extract relevant metadata from .h5 files.
    """
    records = []

    for dirpath, _, files in sorted(os.walk(root_dir)):
        for file in files:
            if not file.endswith(".h5"):
                print(f"Skipping non-HDF5 file: {file}")
                continue

            try:
                slide_id, label_with_ext = file.rsplit("_", 1)
                label = label_with_ext.rsplit(".", 1)[0]
            except ValueError:
                print(f"Unexpected file naming format: {file}")
                continue

            sample_number = sample_number_dict.get(slide_id, "UNKNOWN")

            records.append({
                "patient_id": slide_id,
                "label": label,
                "sample_number": sample_number,
                "tensor_paths": os.path.join(dirpath, file)
            })

    return records

def save_to_csv(records, output_path):
    """
    Save the extracted data to a CSV file.
    """
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Data has been written to {output_path}")

def main():
    ROOT_DIR = "/ictstr01/home/aih/laia.mana/project/DATA/attention_visualization/f_extraction/dinobloom-g/"
    MAPPING_CSV = "/ictstr01/home/aih/laia.mana/project/codes/other_codes/file_to_label.csv"
    OUTPUT_CSV = "/ictstr01/home/aih/laia.mana/project/DATA/attention_visualization/kfold/features_dino.csv" 

    sample_number_dict = load_mapping(MAPPING_CSV)
    records = extract_features(ROOT_DIR, sample_number_dict)
    save_to_csv(records, OUTPUT_CSV)

if __name__ == "__main__":
    main()
