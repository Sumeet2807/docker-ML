import torch
import torch.nn as nn
import torch.nn.functional as nnf



class ANN(nn.Module):
    def __init__(self,in_dim,num_classes):
        super().__init__()
        self.dense1 = nn.Linear(in_dim,128)
        self.dense2 = nn.Linear(128,64)
        self.dense3 = nn.Linear(64,num_classes)

    def forward(self,x):
        x = nnf.relu(self.dense1(x))
        x = nnf.relu(self.dense2(x))
        return(self.dense3(x))


