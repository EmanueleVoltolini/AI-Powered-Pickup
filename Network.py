import torch
import torch.nn as nn
import math
import numpy as np
from contextlib import nullcontext

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
        self.epsilon = 0.0000001
 
    def forward(self, output, target):
        ESR = self.ESR(output,target)
        DC = self.DC(output, target)
        loss = torch.add(ESR,DC)
        return loss + self.epsilon

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
    
    def train_one_epoch(self, in_data, tar_data, up_fr, init_samples, batch_size, optim, loss_func,n_shuffle):
        #shuffle = np.arange(in_data.shape[1])
        shuffle = shuffle_data(in_data.shape[1],n_shuffle)

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