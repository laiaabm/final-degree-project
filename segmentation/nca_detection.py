import os
import argparse
import torch
import cv2  # type: ignore
import numpy as np
from torch.utils import data
from tools import TileDataset, SegNCA, PostProcessor


def load_model(model_path, channel_n=6, hidden_size=128):
    """
    Load the SegNCA model.
    """
    model = SegNCA(channel_n=channel_n, hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def annotate_image(image, points):
    """
    Draw circles on the detected points over the image.
    """
    image_np = image[0].numpy()
    for point in points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image_np, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
    return image_np

def save_annotated_image(image_np, output_path):
    """
    Save the annotated image.
    """
    cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

def main(input_dir, output_dir):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"No valid input dir: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    # Load the model and post-processor
    model = load_model(MODEL_PATH)
    post_processor = PostProcessor(mode="max", level=3)

    # Prepare the dataset
    data_path = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]
    data_name = [file for file in os.listdir(input_dir)]
    dataset = TileDataset(data_path)
    loader = data.DataLoader(dataset, batch_size=1)

    for idx, image in enumerate(loader):
        with torch.no_grad():
            out, _ = model(image, steps=STEPS, fire_rate=0.5)
        points = post_processor.get_coordinates(out)

        annotated_image = annotate_image(image, points)

        output_filename = data_name[idx]
        output_path = os.path.join(output_dir, output_filename)
        save_annotated_image(annotated_image, output_path)

if __name__ == "__main__":
    # Model Parameters
    CHANNEL_N = 6
    STEPS = 6
    MODEL_PATH = "./nca-net/nca_weights"

    parser = argparse.ArgumentParser(description="WBC segmentation and annotation")
    parser.add_argument("--input_dir", required=True, help="Input directory with images")
    parser.add_argument("--output_dir", required=True, help="Output directory to save annotated images")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
