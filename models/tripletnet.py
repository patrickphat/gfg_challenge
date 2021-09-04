import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TripletNet(pl.LightningModule):
    def __init__(self, embeddingnet, use_scheduler:False):
        super().__init__()
        self.embeddingnet = embeddingnet
        self.use_scheduler = use_scheduler
        self.criterion = torch.nn.MarginRankingLoss(0.2)

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x.float())
        embedded_y = self.embeddingnet(y.float())
        embedded_z = self.embeddingnet(z.float())
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        if self.use_scheduler:
            scheduler = ReduceLROnPlateau(optimizer, patience=5)
            return {"optimizer": optimizer, "scheduler": scheduler, "monitor": "train_loss"}
        else: 
            return optimizer

    def forward_loss(self, x,y,z):
        dist_a, dist_b, embedded_x, embedded_y, embedded_z = self.forward(x,y,z)
        target = torch.FloatTensor(dist_a.size()).fill_(1)
        target = Variable(target).to(self.device)
        loss_triplet = self.criterion(dist_a, dist_b, target)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd
        return loss

    def training_step(self, batch, idx):
        x, y, z = batch
        loss = self.forward_loss(x,y,z)
        self.log("train_loss", loss)
        return loss
        
    def validation_step(self, batch, idx):
        x, y, z = batch
        loss = self.forward_loss(x,y,z)
        self.log("val_loss", loss)
        return loss
