#%%
import torch
import Network as net
import pre_processing as pp
import general as gen
import Dataset as data
import librosa

#%%

DATADIR = "ESR audio"
ESR = net.ESRLoss()

arpeggio_tar, fs = librosa.load("Dataset\\anastasia-target.wav", sr=None)
chords_tar, _ = librosa.load("Dataset\chords-target.wav", sr=None)
open_chords_tar, _ = librosa.load("Dataset\open_chords-target.wav", sr=None)
single_note_tar, _ = librosa.load("Dataset\A_blues-target.wav", sr=None)
test_tar, _ = librosa.load("Dataset\mixed_nc-target.wav", sr=None)

arpeggio_in, fs = librosa.load("Dataset\\anastasia-input.wav", sr=None)
chords_in, _ = librosa.load("Dataset\chords-input.wav", sr=None)
open_chords_in, _ = librosa.load("Dataset\open_chords-input.wav", sr=None)
single_note_in, _ = librosa.load("Dataset\A_blues-input.wav", sr=None)
test_in, _ = librosa.load("Dataset\mixed_nc-input.wav", sr=None)

test_tar_split = pp.split_audio_overlap(test_tar, 44100*7,0.75)
arpeggio_tar_split = pp.split_audio_overlap(arpeggio_tar, 44100*7,0.75)
chords_tar_split = pp.split_audio_overlap(chords_tar, 44100*7,0.75)
open_chords_tar_split = pp.split_audio_overlap(open_chords_tar, 44100*7,0.75)
single_note_tar_split = pp.split_audio_overlap(single_note_tar, 44100*7,0.75)

test_in_split = pp.split_audio_overlap(test_in, 44100*7,0.75)
arpeggio_in_split = pp.split_audio_overlap(arpeggio_in, 44100*7,0.75)
chords_in_split = pp.split_audio_overlap(chords_in, 44100*7,0.75)
open_chords_in_split = pp.split_audio_overlap(open_chords_in, 44100*7,0.75)
single_note_in_split = pp.split_audio_overlap(single_note_in, 44100*7,0.75)

#LSTM_arpeggio, fs= librosa.load("ESR audio\LSTM_arpeggio_7sec_96.wav", sr=None)
#LSTM_chords, _ = librosa.load("ESR audio\LSTM_chords_7sec_96.wav", sr=None)
#LSTM_open_chord, _ = librosa.load("ESR audio\LSTM_open_chords_7sec_96.wav", sr=None)
#LSTM_single_note, _ = librosa.load("ESR audio\LSTM_single_note_7sec_96.wav", sr=None)
#LSTM_test, _ = librosa.load("ESR audio\LSTM_test_7sec_96.wav", sr=None)
loss_functions = net.Loss()
network = net.RNN()
best_val_net = torch.load('ESR audio\\5 sec 32 hid')
network.load_state_dict(best_val_net)
test_output, test_loss = network.process_data(chords_in_split,
                                                chords_tar_split, loss_functions, 10)                                               
test_loss_ESR = ESR(test_output, chords_tar_split)
print(test_loss_ESR)