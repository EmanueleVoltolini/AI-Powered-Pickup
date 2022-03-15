# Import all the necessary libraries
import torch
import librosa
import math
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter, freqz

# Signal processing function
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

def smooth_signal(input):
  '''Takes in input a signal and returned a smoothed version of it.
  For the smoothing process is used a blackman window.
  See the function "smooth" for more details.'''
  smooth_input = smooth(input,7,'blackman')
  return smooth_input[3:-3]

def high_pass(input):     
  '''perform an high-pass filter on the signal, 
  now the function is directly implemented in the loss function'''
  out  = signal.lfilter([1 , -0.95],1,input)
  return out

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

def concatenate_audio(input, AUDIO_DIR, ext): 
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
  out[:,0] = normalize_audio(out_in)
  out[:,1] = normalize_audio(out_tar)
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

def normalize_audio(audio):
    max = np.max(np.abs(audio))
    normalized_signal = np.array(audio / max)           # Considering a 16 bit audio 
    return normalized_signal
