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
        shuffle = np.arange(in_data.shape[1])                   #perdona il nome shuffle, in realtà non lo fa ma per comodità ho usato la stessa variabile
        #shuffle = shuffle_data(in_data.shape[1],n_shuffle)

        ep_loss = 0
        # Per capire un minimo di spiego come sono strutturati i dati... 
        # in_data e tar data hanno dimensioni [lunghezza_segmento, numero_segmento, 1], l'ultima dimensione era per gestire file mono (=1)
        # o stereo (=2), in realtà tutti i file usati sono mono, lol.
        for batch_i in range(math.ceil(in_data.shape[1] / batch_size)):
            input_batch = in_data[:,shuffle[batch_i*batch_size:(batch_i+1)*batch_size]]
            target_batch = tar_data[:,shuffle[batch_i*batch_size:(batch_i+1)*batch_size]]
            
            self(input_batch[0:init_samples, :,:])            # initialization of the network weights 
            
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

### questa è la parte del training nel main (manca la parte in cui salvo i valori della loss perchè è tutta da rifare probabilmente)
    
    
    for epoch in range(EPOCHS):   # Epochs è il numero di epochs totale 
        ep_st_time = time.time()  # questa riga serve per i calcoli sul tempo per printare poi quanto tempo ci mette

        # Train one epoch
        epoch_loss = network.train_one_epoch(train_in,train_tar, up_freq, 1000, batch_size,optimiser,loss_functions,n_shuffle) 


# Queste sono le funzioni che uso per preparare i dati (dal load degli audio ai batch)

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
    max = np.max(audio)
    normalized_signal = np.array((audio / np.max(np.abs(audio))) * 32767, np.int16)           # Considering a 16 bit audio 
    return normalized_signal


##############################################################################################################
# Questo è il relativo codice nel main in cui chiamo le funzioni

ext = ["input","target"]        # extension of the audio dataset
AUDIO_DIR = "Dataset"
trainfiles = ["open_chords","hotel_cal","A_blues"]  # name of the audio used for the training data
validfiles = ["anastasia","autumn","funk"]          # name of the audio used for the validation data
testfile = ["mixed_nc"]                             # name of the audio used for the test data

############################################################################################################
######################################### SIGNAL PROCESSING ################################################
############################################################################################################

# pp sta per pre-processing, lo script in cui ci sono tutte le funzioni 

# concatenate the audio to obtain a single audio containing all the data
traindata = pp.concatenate_audio(trainfiles,AUDIO_DIR,ext)           
validata = pp.concatenate_audio(validfiles,AUDIO_DIR,ext)
testdata = pp.concatenate_audio(testfile,AUDIO_DIR,ext)


# Smoothing the input signal given to the network to have a better comparison with the target 
# Questa semplicemente fa uno smoothing e poi faccio un low pass filter (non serve a nulla per la preparazione dei batch)
traindata[:,0] = pp.smooth_signal(traindata[:,0])
validata[:,0] = pp.smooth_signal(validata[:,0])
testdata[:,0] = pp.smooth_signal(testdata[:,0])

traindata[:,0] = pp.butter_lowpass_filter(traindata[:,0], cutoff, fs, order)

# Splitting the audio in overlapping windowed segments that match the input dimension of the network
train_in = pp.split_audio_overlap(traindata[:,0], int(fs*3))
train_tar = pp.split_audio_overlap(traindata[:,1], int(fs*3))

val_in = pp.split_audio_overlap(validata[:,0], int(fs*3))
val_tar = pp.split_audio_overlap(validata[:,1], int(fs*3))

test_in = pp.split_audio_overlap(testdata[:,0],int(fs*3))
test_tar = pp.split_audio_overlap(testdata[:,1],int(fs*3))


network = net.RNN()

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
