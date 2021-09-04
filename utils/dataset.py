from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        class_label = y[idx]


        pos_mask = X[y==class_label]
        neg_mask = Y[y!=class_label]

        pos_idx = np.choice([i for i, x in enumerate(pos_mask) if x])
        neg_idx = np.choice([i for i, x in enumerate(neg_mask) if x])

        anchor = X[idx]
        positive = X[pos_idx]
        negative = X[neg_idx]
        
        return anchor, positive, negative
