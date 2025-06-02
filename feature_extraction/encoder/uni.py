import os
import torch
import timm # type: ignore


def get_model():
    # Model parameters for UNI2-h
    timm_kwargs = {
        'img_size': 224,                # Image size
        'patch_size': 14,               # Patch size
        'depth': 24,                    # Depth of the network
        'num_heads': 24,                # Number of attention heads
        'init_values': 1e-5,            # Initial values for weights
        'embed_dim': 1536,              # Embedding dimension
        'mlp_ratio': 2.66667 * 2,       # MLP ratio
        'num_classes': 0,               # No classes (since this is an encoder)
        'no_embed_class': True,         # No embedding class token
        'mlp_layer': timm.layers.SwiGLUPacked,  # Custom MLP layer
        'act_layer': torch.nn.SiLU,     # SiLU activation function
        'reg_tokens': 8,                # Number of regularization tokens
        'dynamic_img_size': True        # Dynamic image size
    }

    model_weights_path = '/ictstr01/home/aih/laia.mana/project/feature_extraction/fe-net/pytorch_model.bin'
    model = timm.create_model("vit_huge_patch14_224", pretrained=False, **timm_kwargs)
    model.load_state_dict(torch.load(model_weights_path, map_location="cpu"), strict=True)
    return model
