import os
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from tools import TileDataset, SegNCA, PostProcessor
from PIL import Image
import cv2 # type: ignore
import numpy as np

def prepare_output_directory(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def load_dataset(image_dir):
    data_path = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
    data_name = [file for file in os.listdir(image_dir)]
    dataset = TileDataset(data_path)
    loader = data.DataLoader(dataset, batch_size=1)
    return loader, data_name

def load_model(channel_n, steps):
    model = SegNCA(channel_n=channel_n, hidden_size=128)
    model.load_state_dict(torch.load("./pretrained/nca_weights", map_location=torch.device('cpu')))
    model.eval()
    return model

def process_image(image, model, steps, post_processor, idx, data_name, output_dir):
    with torch.no_grad():
        out, _ = model(image, steps=steps, fire_rate=0.5)
    points = post_processor.get_coordinates(out)

    image_np = image[0].numpy()

    for point in points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image_np, (x, y), radius=5, color=(255, 0, 0), thickness=-1)

    # Store the Image
    image_filename = data_name[idx]
    image_path = os.path.join(output_dir, image_filename)
    cv2.imwrite(image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

def main():
    # Configuration
    image_dir = "/ictstr01/home/aih/laia.mana/project/net/"
    OUTPUT_DIR = "/ictstr01/home/aih/laia.mana/project/net_out"
    channel_n = 6
    steps = 6

    # Prepare the output directory
    prepare_output_directory(OUTPUT_DIR)

    # Load dataset and data names
    loader, data_name = load_dataset(image_dir)

    # Load model and post-processor
    model = load_model(channel_n, steps)
    post_processor = PostProcessor(mode="max", level=3)

    # Process each image
    for idx, image in enumerate(loader):
        process_image(image, model, steps, post_processor, idx, data_name, OUTPUT_DIR)

if __name__ == "__main__":
    main()
