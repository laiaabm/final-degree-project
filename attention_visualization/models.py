
############
# ENCODERS
############

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import urllib.request
import timm # type: ignore

def get_encoder_model(encoder):
    if encoder == 'uni2':
        model = uni2()

    elif encoder == 'dinobloom-g':
        model_name="dinov2_vitg14_G"
        model = DinoBloom(model_name)

    ndim = 1536
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])

    return model, transform, ndim



def uni2():
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

    model_weights_path = '/ictstr01/home/aih/laia.mana/project/codes/pretrained_nets/pytorch_model.bin'
    model = timm.create_model("vit_huge_patch14_224", pretrained=False, **timm_kwargs)
    model.load_state_dict(torch.load(model_weights_path, map_location="cpu"), strict=True)
    return model



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
        self.model_path = f"/ictstr01/home/aih/laia.mana/project/codes/pretrained_nets/{model_name}.pth"
        
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
            url = self.model_urls[self.model_name]
            urllib.request.urlretrieve(url, self.model_path)

    def forward(self, x):
        """
        Forward pass to extract DINOv2 patch-level embeddings (x_norm_patchtokens).
        """
        with torch.no_grad():
            #features_dict = self.model.forward_features(x)
            #features = features_dict['x_norm_patchtokens']
            features = self.model(x)
        return features



##############
# AGGREGATORS
##############

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


def get_aggregator_model(method='', ndim=1024, n_classes=2, **kwargs):
    # GMA
    if method == 'AB-MIL':
        return GMA(ndim=ndim, n_classes=n_classes, **kwargs)
    elif method == 'transMIL':
        return TransMIL(ndim=ndim, n_classes=n_classes, **kwargs)
    


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_tasks = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_tasks)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x



class GMA(nn.Module):
    def __init__(self, ndim=1024, gate = True, size_arg = "big", dropout = False, n_classes = 2, n_tasks=1):
        super(GMA, self).__init__()
        self.size_dict = {"small": [ndim, 512, 256], "big": [ndim, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_tasks = 1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1], n_classes)

        initialize_weights(self)

    def get_sign(self, h):
        A, h = self.attention_net(h)# h: Bx512
        w = self.classifier.weight.detach()
        sign = torch.mm(h, w.t())
        return sign

    def forward(self, h, attention_only=False):
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A[0]

        A_raw = A
        w = self.classifier.weight.detach()
        sign = torch.mm(h.detach(), w.t()).cpu().numpy()

        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)

        logits  = self.classifier(M)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'A': A, 'A_raw': A_raw}
        return results_dict

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()



class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2, # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,        # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x, return_attn=False):
        attn_output = self.attn(self.norm(x))
        output = x + attn_output
        if return_attn:
            return output, attn_output
        else:
            return output


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes=2, ndim = 128):
        super(TransMIL, self).__init__()
        dim = round(ndim / 2)
        self.pos_layer = PPEG(dim=dim)
        self._fc1 = nn.Sequential(nn.Linear(ndim, dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=dim)
        self.layer2 = TransLayer(dim=dim)
        self.norm = nn.LayerNorm(dim)
        self._fc2 = nn.Linear(dim, self.n_classes)

    def forward(self, h, **kwargs):
        h = h.unsqueeze(0) # batch_size = 1
        h = self._fc1(h) # [B, n, dim]
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        h, attn_scores1 = self.layer1(h, return_attn=True)
        h = self.pos_layer(h, _H, _W)
        h, attn_scores2 = self.layer2(h, return_attn=True)
        h = self.norm(h)[:, 0]

        # predict
        logits = self._fc2(h)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {
            'logits': logits,
            'Y_prob': Y_prob,
            'Y_hat': Y_hat,
            'attn_scores_layer1': attn_scores1,
            'attn_scores_layer2': attn_scores2
        }
        return results_dict
