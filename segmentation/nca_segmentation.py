import os
import argparse
import torch
import cv2  # type: ignore
import numpy as np
from torch.utils import data
from tools import TileDataset, SegNCA, PostProcessor

def load_model():
    """
    Load SegNCA model
    """
    model = SegNCA(channel_n=CHANNEL_N, hidden_size=128)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def process_image(image, model, post_processor):
    """
    Return the detected points.
    """
    with torch.no_grad():
        out, _ = model(image, steps=STEPS, fire_rate=0.5)
    points = post_processor.get_coordinates(out)
    return points

def crop_cells(points, image, margin):
    """
    Use the detected points to crop the image using a margin.
    """
    image_np = image[0].numpy()
    h, w, _ = image_np.shape

    cropped_cells = []
    for point in points:
        x, y = int(point[0]), int(point[1])

        # Define the limits of the crop
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(w, x + margin), min(h, y + margin)

        # Crop the image
        cropped_cells.append(image_np[y1:y2, x1:x2])

    return cropped_cells

def save_cropped_cells(cropped_cells, image_path, output_dir):
    """
    Save the cropped cells as individual images.
    """
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    for i, cell in enumerate(cropped_cells):
        cell_filename = f"{img_name}_wbc{i}.png"
        cell_path = os.path.join(output_dir, cell_filename)
        cv2.imwrite(cell_path, cv2.cvtColor(cell, cv2.COLOR_RGB2BGR))

def main(input_dir, output_dir):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"No valid input dir: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    # Load the model and the post-processor
    model = load_model()
    post_processor = PostProcessor(mode="max", level=3)

    # Load the dataset
    data_path = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]
    dataset = TileDataset(data_path)
    loader = data.DataLoader(dataset, batch_size=1)

    for image, image_path in loader:
        points = process_image(image, model, post_processor)
        cropped_cells = crop_cells(points, image, MARGIN)

        if cropped_cells:
            save_cropped_cells(cropped_cells, image_path[0], output_dir) 

if __name__ == "__main__":
    # Model Parameters
    CHANNEL_N = 6
    STEPS = 6
    MARGIN = 20
    MODEL_PATH = "./pretrained/nca_weights"

    parser = argparse.ArgumentParser(description="wbc segmentation")
    parser.add_argument("--input_dir", required=True, help="input directory with images")
    parser.add_argument("--output_dir", required=True, help="output directory to save the cropped cells")
    
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)