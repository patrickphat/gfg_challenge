from torch import nn
import pytorch_lightning as pl

class EmbeddingNet(pl.LightningModule):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self,input_dim, dropout=0.25):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)

class EmbeddingNetMedium(pl.LightningModule):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self,input_dim, dropout=0.25):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)

class EmbeddingNetSmall(pl.LightningModule):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self,input_dim, dropout=0.25):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)

class EmbeddingNetTiny(pl.LightningModule):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self,input_dim, dropout=0.25):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)

class EmbeddingNetSuperTiny(pl.LightningModule):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self,input_dim, dropout=0.25):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)
    
class EmbeddingNetAtom(pl.LightningModule):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self,input_dim, dropout=0.25):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)