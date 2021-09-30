from os import path
from pathlib import Path
import torch
from torch import optim
from torch._C import device
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))

model = Model(n_input_features=6)
# train your model...
'''
for param in model.parameters():
    print(param)

FILE = "model.pth"
torch.save(model.state_dict(), FILE)
#model = torch.load(FILE)
#model.eval()


loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

for param in loaded_model.parameters():
    print(param)

print(model.state_dict())
'''
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer.state_dict())

checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}

#torch.save(checkpoint, "checkpoint.pth")
loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optim_state"])

print(optimizer.state_dict())


######SAVE ON GPU AND LOAD ON CPU######
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(),PATH)

device =torch.device('cpu')
model = Model(*args,**kargs)
model.load_state_dict(torch.load(PATH, map_location=device))



#####SAVE ON THE GPU AND LOAD ON THE GPU###########
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(),PATH)

model = Model(*args,**kargs)
model.load_state_dict(torch.load(PATH))
model.to(device)



#######SAVE ON THE CPU AND LOAD ON THE GPU#######
torch.save(model.state_dict(),PATH)

device =torch.device('cuda')
model = Model(*args,**kargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:1"))
model.to(device)