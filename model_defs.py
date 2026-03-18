import random
import numpy as np
import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class PointCloudAE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointCloudAE, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)

        self.dec1 = nn.Linear(self.latent_size,256)
        self.dec2 = nn.Linear(256,512)
        self.dec3 = nn.Linear(512,self.point_size*3)

    def encoder(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x
    
    def decoder(self, x):
       
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)

        return x.view(-1, self.point_size, 3)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Wrapper for Point Transformer v3 + decoder for reconstruction
class PointCloudTransformerAE_fullmodel(nn.Module):
    def __init__(self, encoder, latent_dim=512, num_points=2048):
        super().__init__()
        self.encoder = encoder  # Point Transformer V3
        self.global_pool = nn.AdaptiveMaxPool1d(1)  # Pool per-point features to global
        self.proj = nn.Linear(64, latent_dim)
        self.norm_pooled = nn.LayerNorm(64)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_points * 3)
        )

        self.num_points = num_points

    def forward(self, data_dict):
        point = self.encoder(data_dict)         # Point object with .feat and .coord
        feat = point.feat                       # (B*N, C)
        batch = point.batch                     # (B*N,)
        B = batch.max().item() + 1

        # Aggregate features per batch
        pooled_feats = []
        for i in range(B):
            mask = (batch == i)
            f = feat[mask].unsqueeze(0).transpose(1, 2)  # (1, C, N_i)
            pooled = self.global_pool(f).squeeze(-1)     # (1, C)
            pooled_feats.append(pooled)
        pooled_feats = torch.cat(pooled_feats, dim=0)     # (B, C)
        pooled_feats = self.norm_pooled(pooled_feats)
        latent = self.proj(pooled_feats)  # (B, latent_dim)
        out = self.decoder(latent).view(-1, self.num_points, 3)  # (B, 2048, 3)

        return out