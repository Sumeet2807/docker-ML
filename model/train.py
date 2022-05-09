import torch
from model.ann import ANN 

def train(model):
#dummy training - does nothing     
    pass

net = ANN(64,5)

train(net)

scripted_module = torch.jit.script(net)
torch.jit.save(scripted_module, 'checkpoints/XX_XX.pt')


