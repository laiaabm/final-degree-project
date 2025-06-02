import torch
import torch.nn as nn
import os
import urllib.request

class DinoBloom(nn.Module):
    def __init__(self, model_name="dinov2_vitg14_G"):
        super().__init__()
        
        # Mapping from extended model names to embedding sizes
        # (S, B, L, G correspond to 384, 768, 1024, 1536)
        self.embed_sizes = {
            "dinov2_vits14_S": 384,
            "dinov2_vitb14_B": 768,
            "dinov2_vitl14_L": 1024,
            "dinov2_vitg14_G": 1536
        }
        
        # Mapping from extended model names to official torch.hub names
        self.official_model_names = {
            "dinov2_vits14_S": "dinov2_vits14",
            "dinov2_vitb14_B": "dinov2_vitb14",
            "dinov2_vitl14_L": "dinov2_vitl14",
            "dinov2_vitg14_G": "dinov2_vitg14"
        }
        
        # Mapping from extended model names to their respective download URLs
        self.model_urls = {
            "dinov2_vits14_S": "https://zenodo.org/records/10908163/files/DinoBloom-S.pth?download=1",
            "dinov2_vitb14_B": "https://zenodo.org/records/10908163/files/DinoBloom-B.pth?download=1",
            "dinov2_vitl14_L": "https://zenodo.org/records/10908163/files/DinoBloom-L.pth?download=1",
            "dinov2_vitg14_G": "https://zenodo.org/records/10908163/files/DinoBloom-G.pth?download=1"
        }
        
        # Store model name, infer official name, and set local path
        self.model_name = model_name
        self.official_name = self.official_model_names[model_name]
        self.model_path = f"/ictstr01/home/aih/laia.mana/project/feature_extraction/fe-net/{model_name}.pth"
        
        # Ensure we have the correct .pth file
        self.download_model_if_needed()
        
        # Load the official DINOv2 backbone from torch.hub
        self.model = torch.hub.load('facebookresearch/dinov2', self.official_name)
        
        # Load the fine-tuned state dictionary
        pretrained = torch.load(self.model_path, map_location=torch.device('cpu'))
        
        # Filter out classifier heads (dino_head, ibot_head, etc.) and rename keys
        new_state_dict = {}
        for key, value in pretrained['teacher'].items():
            if 'dino_head' in key or 'ibot_head' in key:
                continue
            # Remove 'backbone.' prefix if it exists
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value
        
        # Update position embedding according to the embedding size for this model
        embed_dim = self.embed_sizes[self.model_name]
        pos_embed = nn.Parameter(torch.zeros(1, 257, embed_dim))
        self.model.pos_embed = pos_embed
        
        # Load the new state dictionary
        self.model.load_state_dict(new_state_dict, strict=True)

    def download_model_if_needed(self):
        """
        Download the DinoBloom checkpoint corresponding to the chosen model_name
        if it doesn't already exist locally.
        """
        if not os.path.exists(self.model_path):
            print(f"Downloading DinoBloom model ({self.model_name}) to {self.model_path}...")
            url = self.model_urls[self.model_name]
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")
        else:
            print(f"DinoBloom model ({self.model_name}) already exists at {self.model_path}.")

    def forward(self, x):
        """
        Forward pass to extract DINOv2 patch-level embeddings (x_norm_patchtokens).
        """
        with torch.no_grad():
            #features_dict = self.model.forward_features(x)
            #features = features_dict['x_norm_patchtokens']
            features = self.model(x)
        return features
