import torch
from torch import nn

class MyNet(nn.Module):
    def __init__(self, n_features : int, task : str = 'classif', n_classes = None):
        """ Blueprint for base Skorch model. """
        
        super(MyNet, self).__init__()
        
        self.n_features = n_features
        self.task = task
        self.n_classes = n_classes
        
        if self.task == 'classif':
            assert self.n_classes is not None & type(self.n_classes) == int , 'Number of classes must be specified \
            if task is a Classification.'
        
        self.layer1 = nn.Linear(self.n_features, 2*self.n_features)
        self.layer2 = nn.Linear(2*self.n_features, 4*self.n_features)
        
        if self.task != 'classif':
            self.layer3 = nn.Linear(4*self.n_features, 1)
        else:
            self.layer3 = nn.Linear(4*self.n_features, self.n_classes)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))

        x = torch.softmax(self.layer3(x)) if self.task == 'classif' else self.layer3(x)
        
        return x