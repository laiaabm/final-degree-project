import os
import glob
import pandas as pd
import h5py
import numpy as np
import torch
import cv2 # type: ignore
import encoders
from PIL import Image
from parser_utils import args_parser


def read_patches(path):
    """
    Reads all .png patches in a given directory and returns them as a list of NumPy arrays.
    """
    patch_data_list = []
    patch_paths = glob.glob(os.path.join(path, '*.png'))
    for patch_path in patch_paths:
        patch_img = Image.open(patch_path)
        # Convert CMYK or RGBA -> RGB if necessary
        if patch_img.mode == 'CMYK':
            patch_img = patch_img.convert('RGB')
        if patch_img.mode == 'RGBA':
            patch_img = patch_img.convert('RGB')
        patch_img = np.array(patch_img)
        patch_data_list.append(patch_img)
    return patch_data_list

def extract_features(
    patch_data_list, slide_name, label, model, transform, ndim, device, args, coords=None
):
    """
    Extract features for all patches for one slide & label; then write an .h5.
    Stores 'label' as well.
    """
    # Create encoder-specific output folder if not exist
    encoder_out_dir = os.path.join(args.out_path, args.encoder)
    os.makedirs(encoder_out_dir, exist_ok=True)
    
    # Prepare the output file path
    out_filename = f"{slide_name}_{label}.h5"
    out_path = os.path.join(encoder_out_dir, out_filename)

    # ---- We already skip the slide in main() if it exists, so no need to skip here. 
    # ---- But you could also keep a second safety check here if you want:
    #
    # if os.path.exists(out_path):
    #     print(f"Features already extracted for {slide_name} with label {label}, skipping.")
    #     return

    feats = np.empty([0, ndim])

    with torch.no_grad():
    #with torch.autocast(device_type="cuda", dtype=torch.float16):
        for patch in patch_data_list:
            # Resize patch
            img = cv2.resize(patch, (args.tilesize, args.tilesize), interpolation=cv2.INTER_CUBIC)
            # Convert from NumPy BGR -> RGB PIL image for transforms
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # Transform
            img_t = transform(img)
            # Add batch dimension
            batch_t = torch.unsqueeze(img_t, 0).to(device)
            # Extract features
            features = model(batch_t)
            # === Average-pool across patch tokens ===
            # shape after mean: (B, EmbDim), e.g. (1, 384)
            #features = features.mean(dim=1) # modified
            features = features.cpu().detach().numpy()
            feats = np.append(feats, features, axis=0)

    print(f"Features size for slide={slide_name}, label={label}: {feats.shape}")

    # Write data to HDF5, also store the label
    with h5py.File(out_path, 'w') as h5f:
        h5f.create_dataset("features", data=feats)
        # Save label as a string in HDF5
        h5f.create_dataset("label", data=np.string_(label))
        # Save coordinates if provided
        if coords is not None:
            h5f.create_dataset("coords", data=coords)

def main():
    parser = args_parser()
    args = parser.parse_args()
    
    # Set up encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, ndim = encoders.get_encoder(args.encoder)
    model.to(device)

    # Load CSV
    metadata = pd.read_csv(args.csv_path)
    # e.g., columns are: slide_id, label

    # List unique slide IDs
    slide_names = metadata['slide_id'].unique()
    print('Unique slide IDs:', len(slide_names))

    # Process each slide
    for slide_name in slide_names:
        # If there's exactly one label per slide, get it:
        subset_df = metadata[metadata['slide_id'] == slide_name]
        if subset_df.empty:
            print(f"No label found for slide {slide_name}, skipping.")
            continue
        label = subset_df['label'].iloc[0]

        # Build path to the folder containing patches
        patches_path = os.path.join(args.data_path, slide_name)
        print(f"Looking for patches in: {patches_path}")

        # ===========================
        # Skip if the output file already exists, BEFORE loading patches
        encoder_out_dir = os.path.join(args.out_path, args.encoder)
        os.makedirs(encoder_out_dir, exist_ok=True)
        out_filename = f"{slide_name}_{label}.h5"
        out_path = os.path.join(encoder_out_dir, out_filename)
        
        if os.path.exists(out_path):
            print(f"Features already extracted for {slide_name} with label {label}, skipping this slide.")
            continue
        # ===========================

        if os.path.exists(patches_path):
            patch_data_list = read_patches(patches_path)
            print(f'Number of patches for {slide_name}: {len(patch_data_list)}')
            extract_features(patch_data_list, slide_name, label,
                             model, transform, ndim, device, args)
        else:
            print(f"Path {patches_path} does not exist; skipping {slide_name}.")

if __name__ == '__main__':
    main()
