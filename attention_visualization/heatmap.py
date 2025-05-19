"""
This script implements attention heatmap visualization for Whole Slide Images (WSIs),
following the workflow presented in the TRIDENT tutorial:
https://github.com/mahmoodlab/TRIDENT/blob/main/tutorials/3-Training-a-WSI-Classification-Model-with-ABMIL-and-Heatmaps.ipynb

Original repository: https://github.com/mahmoodlab/TRIDENT
"""

import os
import torch
import h5py

from models import get_encoder_model, get_aggregator_model

from trident import OpenSlideWSI
from visualization import visualize_heatmap


def main():
    # Get the paths
    svs_path = os.path.join(dataset, slide_name + '.svs')

    # Load the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model, transform, ndim = get_encoder_model(encoder)

    aggregation_model = get_aggregator_model(method, ndim)
    aggregation_model.to(device)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    aggregation_model.load_state_dict(checkpoint['state_dict'])
    aggregation_model.eval()

    # Run inference
    filepath = os.path.join(feature_extraction_path)
    with h5py.File(filepath, 'r') as f:
        patch_features = f['features'][:] 
        coords = f['coords'][:]

    # Get the attention weights
    features_tensor = torch.from_numpy(patch_features).float().to(device)
    results_dict = aggregation_model(features_tensor)

    if method == "AB-MIL":
        A = results_dict['A']
    elif method == "transMIL":
        A = results_dict['attn_scores_layer2']
    else:
        raise ValueError(f"Unknown method: {method}")
    
    A_np = A.detach().cpu().numpy()

    # Load the WSI
    slide = OpenSlideWSI(slide_path=svs_path, lazy_init=False)


    # Get the heatmap
    heatmap = visualize_heatmap(
        wsi=slide,
        scores=A_np,
        coords=coords,
        vis_level=1,
        patch_size_level0=224,
        normalize=True,
        output_dir=heatmap_path,
        output_heatmap=heatmap_name)
    print(heatmap)



if __name__ == "__main__":
    # Define the variables
    encoder = 'dinobloom-g'
    method = 'AB-MIL'

    fe_slide = 'PB AI 0478_chipnegative.h5'
    fe_path = '/ictstr01/home/aih/laia.mana/project/DATA/attention_visualization/f_extraction/dinobloom-g/'
    feature_extraction_path = os.path.join(fe_path, fe_slide)

    checkpoint_path = '/ictstr01/home/aih/laia.mana/project/DATA/aggr_train/tiles_dinobloom-g/AB-MIL/checkpoint_latest_kfold2.pth'

    slide_name = 'PB AI 0478'
    dataset = '/ictstr01/groups/labs/marr/qscd01/datasets/241002_hecker_CHIP/Hips/PB/'

    heatmap_path = '/ictstr01/home/aih/laia.mana/project/DATA/attention_visualization/heatmaps/'
    heatmap_name = f'heatmap_{slide_name}_{encoder}_{method}.png'

    main()