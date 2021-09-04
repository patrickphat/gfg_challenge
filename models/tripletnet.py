import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Variable


class TripletNet(pl.LightningModule):
    def __init__(self, embeddingnet):
        super(TripletNet, self).__init__()
        self.embeddingnet = embeddingnet
        self.criterion = torch.nn.MarginRankingLoss(0.2)

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward_loss(self, x,y,z):
        
        dist_a, dist_b, embedded_x, embedded_y, embedded_z = self.forward(x,y,z)
        target = torch.FloatTensor(dist_a.size()).fill_(1)
        target = Variable(target)
        loss_triplet = criterion(dist_a, dist_b, target)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd
        return loss

    def training_step(self, batch, idx):
        x, y, z = batch
        loss = self.forward_loss(x,y,z)
        return loss
        
    def validation_step(self, batch, idx):
        x, y, z = batch
        loss = self.forward_loss(x,y,z)
        return loss
