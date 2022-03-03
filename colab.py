#%%
# Import all the necessary libraries
import torch
import os
import json
import torch.nn as nn
import torchaudio
import torchvision
import torchvision.transforms as transforms
import time
import wave
import math
import scipy
import librosa
import numpy as np
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from typing import overload
from torch.utils.data import Dataset, dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from numpy import double, pi, log10
from scipy.io import wavfile
from scipy import signal   
from contextlib import nullcontext
from scipy.signal import butter, lfilter, freqz
from scipy.io.wavfile import write


#%% FUNCTIONS

def json_save(data, file_name, dir_name=''):
    dir_name = [dir_name] if ((type(dir_name) != list) and (dir_name)) else dir_name
    assert type(file_name) == str
    file_name = file_name + '.json' if not file_name.endswith('.json') else file_name
    full_path = os.path.join(*dir_name, file_name)
    with open(full_path, 'w') as fp:
        json.dump(data, fp)

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
  '''perform an high-pass filter on the signal, 
  now the function is directly implemented in the loss function'''
  out  = signal.lfilter([1 , -0.95],1,input)
  return out

def concatenate_audio(input): 
  '''Takes in input a list with the name of the audio to concatenate
  and return a single audio with the concatenated audio '''                
  out_in = []
  out_tar = []
  for i in range(len(input)):
    audio_in, fs = librosa.load(AUDIO_DIR + '/' + input[i] + '-' + ext[0] + ".wav", sr=None)
    audio_tar, _ = librosa.load(AUDIO_DIR + '/' + input[i] + '-' + ext[1] + ".wav", sr=None)
    out_in = np.concatenate((out_in,audio_in))
    out_tar = np.concatenate((out_tar,audio_tar))
  out = np.zeros([len(out_in),2])
  out[:,0] = np.array(out_in)
  out[:,1] = np.array(out_tar)
  return out

def split_audio(audio, frame_len): 
  '''Takes in input an audio and splits it in segments of length [frame_len].
  Return a torch.tensor of dimension [frame_len, num_segments,1]'''             
  audio = np.expand_dims(audio, 1) if len(audio.shape) == 1 else audio
  seg_num = math.floor(audio.shape[0] / frame_len)
  dataset = torch.empty((frame_len, seg_num,1))
  # Load the audio for the training set
  for i in range(seg_num):
      dataset[:,i,:] = torch.from_numpy(audio[i * frame_len:(i + 1) * frame_len,:])
      
  return dataset

def split_audio_overlap(audio, frame_len, overlap = 0.5):
  '''Takes in input an audio and splits it in overplapping segments 
  of length [frame_len * overlap], where [overlap] is the percentage of 
  overlapping w.r.t the frame length (std value of overlap is set to 0.5).
  Return a torch.tensor of dimension [frame_len, num_segments,1]'''                                             
  audio = np.expand_dims(audio, 1) if len(audio.shape) == 1 else audio
  seg_num = math.floor(audio.shape[0] / (frame_len * (1 - overlap))) -1
  dataset = torch.empty((frame_len, seg_num,1))
  # Tringular window for the signal 
  triang = torch.tensor(signal.triang(frame_len))

  # Load the audio for the training set
  for i in range(seg_num):
      dataset[:,i,:] = torch.from_numpy(audio[i * int(frame_len*(1 - overlap)):i * int(frame_len*(1 - overlap))+frame_len,:])
      dataset[:,i,0] = torch.mul(triang, dataset[:,i,0])
      
  return dataset

def wrapperargs(func, args):
    '''Takes in input a function and a list of arguments,
    return func(*args)'''
    return func(*args)

def smooth_signal(input):
  '''Takes in input a signal and returned a smoothed version of it.
  For the smoothing process is used a blackman window.
  See the function "smooth" for more details.'''
  smooth_input = smooth(input,7,'blackman')
  return smooth_input[3:-3]

def shuffle_data(input,block):
    num = input%block;
    shuffle = torch.arange(input)
    if (num):
        shuffled = torch.reshape(shuffle[0:-num], (int(len(shuffle)/block),block))
        shuffled = shuffled[torch.randperm(shuffled.size()[0])]
        shuffled = torch.reshape(shuffled, [input-num])
        shuffle[0:-num] = shuffled
    else:
        shuffled = torch.reshape(shuffle[0:], (int(len(shuffle)/block),block))
        shuffled = shuffled[torch.randperm(shuffled.size()[0])]
        shuffled = torch.reshape(shuffled, [input])

    return shuffle

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    y[0:-15] = y[15:]
    y[-15:] = data[-15:]
    return y


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

# DC loss calculates the DC part of the loss function between output/target
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
        
class PreEmphf(nn.Module):
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

# PreEmph performs the high-pass pre-emphasis filter on the signal
class PreEmph(nn.Module):
    def __init__(self):
        super(PreEmph, self).__init__()
        self.epsilon = 0.00001
        self.zPad = len([-0.95, 1]) - 1

        self.conv_filter = nn.Conv1d(1, 1, 2, bias=False)
        self.conv_filter.weight.data = torch.tensor([[[-0.95, 1]]], requires_grad=False)
        
    def forward(self, output, target):
      # zero pad the input/target so the filtered signal is the same length
      output = torch.cat((torch.zeros(self.zPad, output.shape[1], 1), output))
      target = torch.cat((torch.zeros(self.zPad, target.shape[1], 1), target))
      # Apply pre-emph filter, permute because the dimension order is different for RNNs and Convs in pytorch...
      output = self.conv_filter(output.permute(1, 2, 0))
      target = self.conv_filter(target.permute(1, 2, 0))

      return output.permute(2, 0, 1), target.permute(2, 0, 1)

# DC loss calculates the DC part of the loss function between output/target
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

# ESR loss calculates the Error-to-signal between the output/target
class ESRLoss(nn.Module):
    def __init__(self):
        super(ESRLoss,self).__init__()
        self.pre = PreEmph()
        self.epsilon = 0.00001
 
    def forward(self, output, target):
        output , target = self.pre(output, target)
        loss = torch.add(target, -output)
        loss = torch.pow(loss, 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss

# Loss calculate the loss function summing the 2 contributions (ESR + DC)
class Loss(nn.Module):
    def __init__(self):
        super(Loss,self).__init__()
        self.ESR = ESRLoss()
        self.DC = DCLoss()
        self.epsilon = 0.00001
 
    def forward(self, output, target):
        ESR = self.ESR(output,target)
        DC = self.DC(output, target)
        loss = torch.add(ESR,DC)
        return loss

# Main class for the LSTM RNN
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
        shuffle = np.arange(in_data.shape[1])
        #shuffle = shuffle_data(in_data.shape[1],7)

        ep_loss = 0

        for batch_i in range(math.ceil(in_data.shape[1] / batch_size)):
            input_batch = in_data[:,shuffle[batch_i*batch_size:(batch_i+1)*batch_size]]
            target_batch = tar_data[:,shuffle[batch_i*batch_size:(batch_i+1)*batch_size]]
            
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
            #ramp = np.ones((200,1,1))
            #ramp[:,0,0] = np.linspace(0,1,200)
            #ramp = torch.from_numpy(ramp).cuda()
            #output[0:200] = torch.mul(ramp,output[0:200]) 
            # If the data set doesn't divide evenly into the chunk length, process the remainder
            if not (output.size()[0] / chunk).is_integer():
                output[(l + 1) * chunk:-1] = self(input_data[(l + 1) * chunk:-1])
            self.reset_hidden()
            loss = loss_fcn.forward(output, target_data)
            
        return output, loss

class TrainTrack(dict):
    def __init__(self):
        self.update({'current_epoch': 0, 'training_losses': [], 'validation_losses': [], 'train_av_time': 0.0,
                    'val_av_time': 0.0, 'total_time': 0.0, 'best_val_loss': 1e12, 'test_loss': 0})

    def restore_data(self, training_info):
        self.update(training_info)

    def train_epoch_update(self, loss, ep_st_time, ep_end_time, init_time, current_ep):
        if self['train_av_time']:
            self['train_av_time'] = (self['train_av_time'] + ep_end_time - ep_st_time) / 2
        else:
            self['train_av_time'] = ep_end_time - ep_st_time
        self['training_losses'].append(loss)
        self['current_epoch'] = current_ep
        self['total_time'] += ((init_time + ep_end_time - ep_st_time)/3600)

    def val_epoch_update(self, loss, ep_st_time, ep_end_time):
        if self['val_av_time']:
            self['val_av_time'] = (self['val_av_time'] + ep_end_time - ep_st_time) / 2
        else:
            self['val_av_time'] = ep_end_time - ep_st_time
        self['validation_losses'].append(loss)
        if loss < self['best_val_loss']:
            self['best_val_loss'] = loss

#%%

if __name__ == "__main__":
    start_time = time.time()

    ############################################################################################################
    ############################################# PARAMETERS ###################################################
    ############################################################################################################
    EPOCHS = 1
    LEARNING_RATE = 5*pow(10, -4)
    up_freq = 2048
    AUDIO_DIR = "/content/Dataset"
    save_path = "/content/Result"
    PATH = 'AI-0' 
    n_segments = 40
    ext = ["input","target"]        #extension of the audio dataset
    fs = 44100
    batch_size = 40
    model_name = "RNN"

    # Filter requirements
    order = 6
    fs = 44100                                          # sample rate, Hz
    cutoff = 2000                                       # desired cutoff frequency of the filter, Hz
    overlap = 0.5
    validation_f = 1    	                              # validation frequency in number of epochs 
    validation_p = 500                                  # validation patient in number of epochs

    trainfiles = ["open_chords","hotel_cal","A_blues"]  # name of the audio used for the training data
    validfiles = ["anastasia","autumn","funk"]          # name of the audio used for the validation data
    testfile = ["mixed_nc"]                             # name of the audio used for the test data

    ############################################################################################################
    ######################################### SIGNAL PROCESSING ################################################
    ############################################################################################################

    # concatenate the audio to obtain a single audio containing all the data
    traindata = concatenate_audio(trainfiles)           
    validata = concatenate_audio(validfiles)
    testdata = concatenate_audio(testfile)
  

    # Smoothing the input signal given to the network to have a better comparison with the target 
    traindata[:,0] = smooth_signal(traindata[:,0])
    validata[:,0] = smooth_signal(validata[:,0])
    testdata[:,0] = smooth_signal(testdata[:,0])
    
    traindata[:,0] = butter_lowpass_filter(traindata[:,0], cutoff, fs, order)
    
    '''
    train_hp = np.zeros(traindata.shape)
    val_hp = np.zeros(validata.shape)
    test_hp = np.zeros(testdata.shape)

    for i in range(2):
        train_hp[:,i] = high_pass(traindata[:,i])
        val_hp[:,i] = high_pass(validata[:,i])
        test_hp[:,i] = high_pass(testdata[:,i])
    '''
    # Splitting the audio in overlapping windowed segments that match the input dimension of the network
    train_in = split_audio_overlap(traindata[:,0], int(fs*3),0.5)
    train_tar = split_audio_overlap(traindata[:,1], int(fs*3),0.5)

    #val_in = split_audio_overlap(validata[:,0], int(fs*3), 0.25)
    #val_tar = split_audio_overlap(validata[:,1], int(fs*3), 0.25)
    val_in = split_audio_overlap(validata[:,0], int(fs*3))
    val_tar = split_audio_overlap(validata[:,1], int(fs*3))

    test_in = split_audio_overlap(testdata[:,0],int(fs*3))
    test_tar = split_audio_overlap(testdata[:,1],int(fs*3))

    #train_tar [:,:,0] = 0
    network = RNN()
    
    # Check if a cuda device is available
    if not torch.cuda.is_available():
        print('cuda device not available/not selected')
        cuda = 0
    else:
        # set all the variable on the GPU
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

    
    # Defining the used optimizer
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Defining the loss function
    loss_functions = Loss()   
    train_track = TrainTrack()
    
    # Defining the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.5, patience=5, verbose=True)

    # Initialization of other parameters 
    init_time = time.time() - start_time + train_track['total_time']*3600
    patience_counter = 0
    writer = SummaryWriter(os.path.join('runs2', model_name))

#%%  TRAINING
    losses = []                         #list to save the loss value each epoch
    min_loss = 1e10
    
    for epoch in range(EPOCHS):
        ep_st_time = time.time()
        print(f"Epoch {epoch+1}")

        # Train one epoch
        epoch_loss = network.train_one_epoch(train_in,train_tar, up_freq, 1000, batch_size,optimiser,loss_functions)
        
        # Append the value of the loss at each epoch
        losses.append(epoch_loss.item())
        
        if (epoch_loss.item() < min_loss):
            print("Alè!")
            count = 0
            min_loss = epoch_loss.item()
        else:
            count = count + 1

        # Run validation
        if epoch % validation_f == 0:
            val_ep_st_time = time.time()
            val_output, val_loss = network.process_data(val_in,
                                             val_tar, loss_functions, 10)
            scheduler.step(val_loss)
            if val_loss < train_track['best_val_loss']:
                patience_counter = 0
                torch.save(network.state_dict(),os.path.join(save_path,'model_best'))
                write(os.path.join(save_path, "best_val_out.wav"),
                      fs, val_output.cpu().numpy()[:, 0, 0])
            else:
                patience_counter += 1
            train_track.val_epoch_update(val_loss.item(), val_ep_st_time, time.time())
            writer.add_scalar('Loss/val', train_track['validation_losses'][-1], epoch)

        print('current learning rate: ' + str(optimiser.param_groups[0]['lr']))
        train_track.train_epoch_update(epoch_loss.item(), ep_st_time, time.time(), init_time, epoch)
        writer.add_scalar('Loss/train', train_track['training_losses'][-1], epoch)
        writer.add_scalar('LR/current', optimiser.param_groups[0]['lr'])

        if validation_p and patience_counter > validation_p:
            print('validation patience limit reached at epoch ' + str(epoch))
            break
    lossESR = ESRLoss()
    test_output, test_loss = network.process_data(test_in,
                                     test_tar, loss_functions, 10)
    test_loss_ESR = lossESR(test_output, test_tar)
    write(os.path.join(save_path, "test_out_final.wav"), fs, test_output.cpu().numpy()[:, 0, 0])
    writer.add_scalar('Loss/test_loss', test_loss.item(), 1)
    writer.add_scalar('Loss/test_lossESR', test_loss_ESR.item(), 1)
    train_track['test_loss_final'] = test_loss.item()
    train_track['test_lossESR_final'] = test_loss_ESR.item()

    
    best_val_net = torch.load(os.path.join(save_path,'model_best'))
    network.load_state_dict(best_val_net)
    test_output, test_loss = network.process_data(test_in,
                                     test_tar, loss_functions, 10)
    test_loss_ESR = lossESR(test_output, test_tar)
    write(os.path.join(save_path, "test_out_bestv.wav"),
          fs, test_output.cpu().numpy()[:, 0, 0])
    writer.add_scalar('Loss/test_loss', test_loss.item(), 2)
    writer.add_scalar('Loss/test_lossESR', test_loss_ESR.item(), 2)
    train_track['test_loss_best'] = test_loss.item()
    train_track['test_lossESR_best'] = test_loss_ESR.item()
    json_save(train_track, 'training_stats', save_path)
    
    '''
#%% Plot the loss results
    # array to describe the number of epoch
    x = np.linspace(1,len(losses),len(losses))          
    
    plt.figure()
    plt.plot(x,losses)
    plt.xlabel("epochs")
    plt.title("loss")  
    '''


