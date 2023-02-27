import torch
import numpy as np

class Dataset:
    def __init__(self, df):
        self.features = torch.tensor(np.stack(df['numerized_tweet']))
        self.targets = torch.tensor(np.asarray(df['Toxicity'])).unsqueeze(1).float()

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        features = self.features[idx]
        target = self.targets[idx]
        return features, target