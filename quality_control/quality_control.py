"""
This Python script is modified based on the work available at https://github.com/josegcpa/haemorasis/blob/main/scripts/python/quality_control.py 
Original repository: https://github.com/josegcpa/haemorasis/tree/main 

Predicts which tiles are of good quality from a folder of tiles (PNG format)
and saves the good-quality images in an output folder.
"""

import argparse
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import os
from glob import glob
from PIL import Image
import numpy as np
import shutil

def read_tiles(tiles_folder):
    '''
    Generator that reads PNG tiles from a directory and normalizes pixel values.
    '''
    tile_files = glob(os.path.join(tiles_folder, "*.png"))
    for tile_path in tile_files:
        image = Image.open(tile_path)
        image = np.array(image) / 255.0 # Normalize [0, 1]
        yield image, tile_path

def main(tiles_folder, checkpoint_path, output_folder, batch_size):
    quality_net = keras.models.load_model(checkpoint_path) # Pretrained model
    os.makedirs(output_folder, exist_ok=True)

    # Define the shapes and output types for the TensorFlow dataset
    output_types = (tf.float32, tf.string)
    output_shapes = ([None, None, n_channels], []) # No fixed height or width
    tf_dataset = tf.data.Dataset.from_generator(
        lambda: read_tiles(tiles_folder),
        output_types=output_types,
        output_shapes=output_shapes)
    tf_dataset = tf_dataset.batch(batch_size, drop_remainder=False)
    tf_dataset = tf_dataset.prefetch(5)

    # Evaluate tile quality and save good-quality images
    for image_batch, path_batch in tqdm(tf_dataset):
        predictions = quality_net(image_batch)
        for path, pred in zip(path_batch.numpy(), predictions.numpy()):
            tile_name = path.decode("utf-8")
            pred_value = float(pred[0]) if pred.ndim > 0 else float(pred)
            quality_label = int(pred_value > 0.5)
            # 1 = good quality
            # 0 = bad quality 
            print(f"TILE: {tile_name}")
            print(f"QUALITY: {quality_label}")
            print(f"PRED VALUE: {pred_value:.4f} \n \n")

            # Copy good quality tiles to the output folder
            if quality_label == 1:
                shutil.copy(tile_name, os.path.join(output_folder, os.path.basename(tile_name)))

if __name__ == "__main__":
    n_channels = 3
    parser = argparse.ArgumentParser(description='Predicts which tiles are of good quality from a folder of PNG tiles and saves them.')
    parser.add_argument('--tiles_folder', dest='tiles_folder', action='store', type=str, required=True, help="Path to folder containing the tiles.")
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', action='store', type=str, required=True, help='Path to checkpoint.')
    parser.add_argument('--output_folder', dest='output_folder', action='store', type=str, required=True, help='Path to folder where good-quality tiles will be saved.')
    parser.add_argument('--batch_size', dest='batch_size', action='store', type=int, required=True, help='Size of mini batch.')
    args = parser.parse_args()
    main(args.tiles_folder, args.checkpoint_path, args.output_folder, args.batch_size)
