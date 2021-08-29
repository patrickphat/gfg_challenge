from torch import nn

class EmbeddingNet(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self,input_dim):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(input_dim, 64),
          nn.ReLU(),
          nn.Linear(64, 128),
          nn.ReLU(),
          nn.Linear(128, 64)
        )
    def forward(self, x):
        '''
        Forward pass
        '''
        return self.layers(x)
