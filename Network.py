#%%
# Import all the necessary libraries
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
import torchvision
import torchvision.transforms as transforms
import wave
import math
import scipy
import librosa
import numpy as np
import librosa.display
from numpy import pi, log10
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from scipy.io import wavfile
from scipy import signal   
from pydub import AudioSegment, effects  
from contextlib import nullcontext

#%% FUNCTIONS

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def high_pass(input):
  out  = signal.lfilter([1 , -0.95],1,input)
  return out

def concatenate_audio(input):
  out_in = []
  out_tar = []
  for i in range(len(input)):
    fs, audio_in = wavfile.read(AUDIO_DIR + '/' + input[i] + '-' + ext[0] + ".wav")
    _, audio_tar = wavfile.read(AUDIO_DIR + '/' + input[i] + '-' + ext[1] + ".wav")
    out_in = np.concatenate((out_in,audio_in))
    out_tar = np.concatenate((out_tar,audio_tar))
  out = np.zeros([len(out_in),2])
  out[:,0] = np.array(out_in)
  out[:,1] = np.array(out_tar)
  return out

def split_audio(audio, frame_len):
  audio = np.expand_dims(audio, 1) if len(audio.shape) == 1 else audio
  seg_num = math.floor(audio.shape[0] / frame_len)
  dataset = torch.empty((frame_len, seg_num,1))
  # Load the audio for the training set
  for i in range(seg_num):
      dataset[:,i,:] = torch.from_numpy(audio[i * frame_len:(i + 1) * frame_len,:])
      
  return dataset

def wrapperargs(func, args):
    return func(*args)

def smooth_signal(input):
  smooth_input = smooth(input,7,'blackman')
  return smooth_input[3:-3]

#%% MODEL 

# ESR loss calculates the Error-to-signal between the output/target
class ESRLoss(nn.Module):
    def __init__(self):
        super(ESRLoss, self).__init__()
        self.epsilon = 0.001

    def forward(self, output, target):
        loss = torch.add(target, -output)
        loss = torch.pow(loss, 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss

class DCLoss(nn.Module):
    def __init__(self):
        super(DCLoss, self).__init__()
        self.epsilon = 0.00001

    def forward(self, output, target):
        loss = torch.pow(torch.add(torch.mean(target, 0), -torch.mean(output, 0)), 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss
        
class PreEmph(nn.Module):
    def __init__(self, filter_cfs, low_pass=0):
        super(PreEmph, self).__init__()
        self.epsilon = 0.00001
        self.zPad = len(filter_cfs) - 1

        self.conv_filter = nn.Conv1d(1, 1, 2, bias=False)
        self.conv_filter.weight.data = torch.tensor([[filter_cfs]], requires_grad=False)

        self.low_pass = low_pass
        if self.low_pass:
            self.lp_filter = nn.Conv1d(1, 1, 2, bias=False)
            self.lp_filter.weight.data = torch.tensor([[[0.85, 1]]], requires_grad=False)

    def forward(self, output, target):
        # zero pad the input/target so the filtered signal is the same length
        output = torch.cat((torch.zeros(self.zPad, output.shape[1], 1), output))
        target = torch.cat((torch.zeros(self.zPad, target.shape[1], 1), target))
        # Apply pre-emph filter, permute because the dimension order is different for RNNs and Convs in pytorch...
        output = self.conv_filter(output.permute(1, 2, 0))
        target = self.conv_filter(target.permute(1, 2, 0))

        if self.low_pass:
            output = self.lp_filter(output)
            target = self.lp_filter(target)

        return output.permute(2, 0, 1), target.permute(2, 0, 1)

class LossWrapper(nn.Module):
    def __init__(self, losses, pre_filt=None):
        super(LossWrapper, self).__init__()
        loss_dict = {'ESR': ESRLoss(), 'DC': DCLoss()}
        if pre_filt:
            pre_filt = PreEmph(pre_filt)
            loss_dict['ESRPre'] = lambda output, target: loss_dict['ESR'].forward(*pre_filt(output, target))
        loss_functions = [[loss_dict[key], value] for key, value in losses.items()]

        self.loss_functions = tuple([items[0] for items in loss_functions])
        try:
            self.loss_factors = tuple(torch.Tensor([items[1] for items in loss_functions]))
        except IndexError:
            self.loss_factors = torch.ones(len(self.loss_functions))

    def forward(self, output, target):
        loss = 0
        for i, losses in enumerate(self.loss_functions):
            loss += torch.mul(losses(output, target), self.loss_factors[i])
        return loss

class Loss(nn.Module):
    def __init__(self):
        self.epsilon = 0.00001
 
    def forward(self, output, target):
        loss = torch.add(target, -output)
        loss = torch.pow(loss, 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss


class RNN(nn.Module):

    def __init__(self, input_size=1, output_size=1, hidden_size=96, num_layers=1,bias=True):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.LSTM = nn.LSTM(input_size,hidden_size,num_layers,bias)
        self.FullyConnected = nn.Linear(hidden_size,output_size,bias)
        self.hidden = None

    def forward(self, x):
        res = x[:,:,0:1]
        x, self.hidden = self.LSTM(x, self.hidden)        
        return self.FullyConnected(x) + res

    # detach hidden state, this resets gradient tracking on the hidden state
    def detach_hidden(self):
        if self.hidden.__class__ == tuple:
            self.hidden = tuple([h.clone().detach() for h in self.hidden])
        else:
            self.hidden = self.hidden.clone().detach()
    
    # changes the hidden state to None, causing pytorch to create an all-zero hidden state when the rec unit is called
    def reset_hidden(self):
        self.hidden = None
    
    def train_one_epoch(self, in_data, tar_data, up_fr, init_samples, batch_size, optim, loss_func):
        shuffle = torch.randperm(in_data.shape[1])

        ep_loss = 0

        for batch_i in range(math.ceil(shuffle.shape[0] / batch_size)):
            input_batch = in_data[:,shuffle[batch_i*batch_size:(batch_i+1)*batch_size],:]
            target_batch = tar_data[:,shuffle[batch_i*batch_size:(batch_i+1)*batch_size],:]
            
            self(input_batch[0:init_samples, :,:])
            
            self.zero_grad()                                  #set the gradient to zero, we don't want the gradient to acccumulate for each mini-batch

            start_i = init_samples
            batch_loss = 0
            for k in range(math.ceil((input_batch.shape[0] - init_samples) / up_fr)):
              output = self(input_batch[start_i:start_i + up_fr, :,:])

              loss = loss_func.forward(output,target_batch[start_i:start_i + up_fr, :,:])
              loss.backward()
              optim.step()

              # Set the network hidden state, to detach it from the computation graph
              self.detach_hidden()
              self.zero_grad()

              # Update the start index for the next iteration and add the loss to the batch_loss total
              start_i += up_fr
              batch_loss += loss

            # Add the average batch loss to the epoch loss and reset the hidden states to zeros
            ep_loss += batch_loss / (k + 1)
            self.reset_hidden()
        return ep_loss / (batch_i + 1)

    # only proc processes a the input data and calculates the loss, optionally grad can be tracked or not
    def process_data(self, input_data, target_data, loss_fcn, chunk, grad=False):
        with (torch.no_grad() if not grad else nullcontext()):
            output = torch.empty_like(target_data)
            for l in range(int(output.size()[0] / chunk)):
                output[l * chunk:(l + 1) * chunk] = self(input_data[l * chunk:(l + 1) * chunk])
                self.detach_hidden()
            # If the data set doesn't divide evenly into the chunk length, process the remainder
            if not (output.size()[0] / chunk).is_integer():
                output[(l + 1) * chunk:-1] = self(input_data[(l + 1) * chunk:-1])
            self.reset_hidden()
            loss = loss_fcn.forward(output, target_data)
        return output, loss

#%%
if __name__ == "__main__":

    # Define some constant
    EPOCHS = 10
    LEARNING_RATE = 5*pow(10, -4)
    #LEARNING_RATE = 0.01
    up_freq = 2048
    AUDIO_DIR = "Dataset"
    n_segments = 40
    ext = ["input","target"]
    fs = 44100
    batch_size = 40

    trainfiles = ["chords","hotel_cal","A_blues"]
    validfiles = ["anastasia","autumn","funk"]
    testfile = ["mix_note_chord"]

    traindata = concatenate_audio(trainfiles)
    validata = concatenate_audio(validfiles)
    testdata = concatenate_audio(testfile)

    #train_split_in = split_audio(traindata[:,0],int(fs/2))
    #train_split_tar = split_audio(traindata[:,1],int(fs/2))
 
    train_split_in = split_audio(traindata[:,0], int(fs/2))
    train_split_tar = split_audio(traindata[:,1], int(fs/2))

    traindata[:,0] = smooth_signal(traindata[:,0])
    validata[:,0] = smooth_signal(validata[:,0])
    testdata[:,0] = smooth_signal(testdata[:,0])
    

    train_hp = np.zeros(traindata.shape)
    val_hp = np.zeros(validata.shape)
    test_hp = np.zeros(testdata.shape)

    for i in range(2):
        train_hp[:,i] = high_pass(traindata[:,i])
        val_hp[:,i] = high_pass(validata[:,i])
        test_hp[:,i] = high_pass(testdata[:,i])

    train_in = split_audio(train_hp[:,0], int(fs/2))
    train_tar = split_audio(train_hp[:,1], int(fs/2))

    val_in = split_audio(val_hp[:,0], int(fs/2))
    val_tar = split_audio(val_hp[:,1], int(fs/2))

    test_in = torch.tensor(testdata[:,0])
    test_tar = torch.tensor(testdata[:,1])

    network = RNN()
    
    # Check if a cuda device is available
    if not torch.cuda.is_available():
        print('cuda device not available/not selected')
        cuda = 0
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(0)
        print('cuda device available')
        network = network.cuda()
        train_in = train_in.cuda()
        train_tar = train_tar.cuda()
        val_in = val_in.cuda()
        val_tar = val_tar.cuda()
        test_in = test_in.cuda()
        test_tar = test_tar.cuda()
        cuda = 1

    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    #loss_functions = LossWrapper({"ESR": 0.75})
    loss_functions = Loss()   
    

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.5, patience=5, verbose=True)

#%%  TRAINING'
    losses = []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}")
        epoch_loss = network.train_one_epoch(train_in,train_tar, up_freq, 1000, batch_size,optimiser,loss_functions)
        losses.append(epoch_loss.item())
        # Run validation
#        val_output, val_loss = network.process_data(val_in,
#                                            val_tar, loss_functions,7)
#        scheduler.step(val_loss)
#%%
    x = np.linspace(1,len(losses),10)
    plt.figure()
    plt.plot(x,losses)
    plt.xlabel("epochs")
    plt.title("loss")  
# %%
    PATH = 'AI-Powered-Pickup_2'
    torch.save(network.state_dict(),PATH)
#%% LOAD A MODEL
    network = RNN()
    network.load_state_dict(torch.load(PATH))
    network.eval()
# %%
    train_split_in = split_audio(traindata[:,0], int(fs/2))
    train_split_tar = split_audio(traindata[:,1], int(fs/2))
#    output, loss = network.process_data(train_split_in,
#                                     train_split_tar, loss_functions, 100000)
#    from scipy.io.wavfile import write
#    write("test_out_final.wav", 44100, output.cpu().numpy()[:, 0, 0].astype(np.int16))
# %%
    output, loss = network.process_data(train_split_in,
                                     train_split_tar, loss_functions, 1)

#%%
    out = np.zeros([len(traindata),1])
    order = output[1]
    for i in range(output.shape[1]):
        out[i*output.shape[0]:i*output.shape[0]+output.shape[0]] = (output[:,i,:]).cpu().numpy()

#%%
    from scipy.io.wavfile import write
    write("test_out_final.wav", 44100, out.astype(np.int16))
# %%
