# Import all the necessary libraries
import torch
import os
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torchaudio
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import signal_proc as sign

# Define some constant
EPOCHS = 2
LEARNING_RATE = .001
n = 1000

# Allocate some variables 
eps_h = np.zeros([1000,])

def loss_func(y_hi, y_ht):
    num_h = 0
    den_h = 0
    for j in range(n):
        num_h = num_h + (abs(y_ht[j]-y_hi[j]))**2
        den_h = den_h + (abs(y_ht[j]))**2
    eps_h = num_h/den_h
    return eps_h
class GuitarDataset(Dataset):

    def __init__(self, audio_dir):
        self.audio_dir = audio_dir

    def __len__(self):
        pass

    def __getitem__(self, name, type):
        audio_sample_path  = os.path.join(self.audio_dir,name + "-" + type + ".wav")
        print(audio_sample_path)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, sr


class RNN(nn.Module):

    def __init__(self, input_size=1, output_size=1, hidden_size=96, num_layers=1,bias=True):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.LSTM = nn.LSTM(input_size,hidden_size,num_layers,bias)
        self.FullyConnected = nn.Linear(hidden_size,output_size,bias)

    def forward(self, x):
        res = x
        x_LSTM = self.LSTM(x)
        predicted = self.FullyConnected(x_LSTM) + res
        return predicted

# Need to modify according to my code
def train_one_epoch(model, data_loader, optimizer, device):
    for inputs, targets in data_loader:
        imputs, targets = inputs.to(device), targets.to(device)         # We need to assign the data to the device

        inputs = sign.high_pass(imputs)
        targets = sign.high_pass(targets)
        # calculate Loss
        predictions  = model(inputs)
        loss = loss_func(predictions, targets)

        # backpropagate loss and update weights
        optimizer.zero_grad()                                           # at every iteration the optimizer is gonna calculate gradients to update the weights, these gradients
                                                                        # are saved. At each iteration we want to reset the gradients to zero, so we can start from scratch
        loss.backward()                                                 # apply back propagation
        optimizer.step()                                                # update the weights

    print(f'Loss: {loss.item()}')

def train(model, data_loader, loss_fn, optimizer, device, epochs):      # higher level function that will use train_one_epoch at each iteration,
                                                                        # we will go throught all the epoch that we want to train the model for
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("------------------")
    print("Train is done")
#


AUDIO_DIR = "Dataset"
Guitar_dataset = GuitarDataset(AUDIO_DIR)
audio_input , sample_rate = Guitar_dataset.__getitem__("Chords","input")    
print(audio_input)
print(sample_rate)
audio_target, sample_rate_tar = Guitar_dataset.__getitem__("Chords","target")
print(audio_target)
print(sample_rate_tar)

GuitarRNN = RNN()
print(GuitarRNN)