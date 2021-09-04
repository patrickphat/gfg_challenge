from torch.utils.data import Dataset
import random

class TripletDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        class_label = self.y[idx]


        pos_mask = self.y==class_label
        neg_mask = self.y!=class_label

        pos_idx = random.choice([i for i, x in enumerate(pos_mask) if x])
        neg_idx = random.choice([i for i, x in enumerate(neg_mask) if x])

        anchor = self.X[idx]
        positive = self.X[pos_idx]
        negative = self.X[neg_idx]
        
        return anchor, negative, positive
    
    def __len__(self):
        return self.X.shape[0] 
