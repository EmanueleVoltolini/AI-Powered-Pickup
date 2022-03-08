# ***LOGBOOK***

## --> 01/10/2021

## **Link to the presentation**

- Presentation of the thesis at the reasearch group:
  - <https://docs.google.com/presentation/d/15ElxQTOWF413CEzKa1X-jkytDLCtqlsn4-r6IHI-xv8/edit#slide=id.gf080f14cc4_0_6>

## **Done**

- Find the source code of the network discussed in the reference paper
- Start to record sound from acoustic guitar with piezo pickup simultaneously with a SM57 mic using a Roland audio interface
- Start learning the basics of pytorch framework in order to have a better comprehension of the paper code and start writing my own
- Read correlated paper on how organize the dataset

## **Problems encountered**

- Learning the Pytorch framework takes a lot of time
- Training the network takes a lot of computational power
- I haven’t plan well how the dataset will be structured

## **Next steps**

- Continue to record guitar samples for the construction of the dataset
- Continue to experiment with the Pytorch framework
- Start to build a new network from scratch, having as a reference the RNN network in the paper code
- Start working full time on the thesis (certainly from 15 October)

## ***CONTENT***

## *Network*

![alt text](network_structure.png "Network Structure")

- Input size: 1
- LSTM layer: 32|64|96 hidden size
- Fully Connected layer : 1 neuron
- Output size: 1
- Residual connection
- Adam optimizer

## *Dataset*

![alt text](dataset_len.png "Dataset Length")

The image is taken from the reference paper, we can try to structure our dataset as:

- Short guitar samples (less than 1 min)
- Concatenate samples to reach more than 3 min for the training data
- Different style of guitar playing
- Different guitars
- Different microphones

## --> OCTOBER 2021

## **Done**

- understand how works the loss function of the reference paper and see if it's meaningful for our purpose
- start to write the first model of the network
- record the first audio for the dataset
- understand the real-time application of the code
- update the bibliography
- design the filters for the loss function
- start calculationg the loss function in matlab
- plot and testing the loss function
- evaluate the loss function with the "fake" signal:
  - signal + filtered white noise  
  - signal + sinusoids
- study the spectrograms of the input and target
- find a smoothing algorithm for the input signal to resolve the problem of the energy at the high frequencies in the input
- write the loss function in python
- write the low and high pass filters in python
- test on the loss function

## ***DATASET***

![alt text](dataset.png "Dataset")

Guitar has been recorded with piezo and SM57 microphone simultaneously.
The signals pass through a Roland audio interface with sample frequency of 44.1 kHz, the output is fed into Ableton and extracted as a mono audio track

## ***STUDY OF THE SIGNALS***

![alt text](spectrogram_oct.png "Spectrogram_oct")

- Input has way more power in the upper register  → smoothing algorithm

![alt text](spectrogram_oct_smooth.png "Spectrogram_oct_smooth")

Here is the difference btw the target sound and smoothed input:

![alt text](spec_diff_oct.png "Diff_spectrogram_oct")

As we can se the major differences between the two signals are in the low frequencies, so we thought to apply a low pass filter on the smoothed input signal.

## ***LOSS FUNCTION***

![alt text](loss_function.png "Loss Function")

These are the filters used:

![alt text](Filters.png "Filters")

The aim of the test performed on the loss function is to see if the loss used in the reference paper suits also our purpose. We take the recoded audio and calculate the loss of different segments, we also create a "fake" signal summing sinusoids at different frequencies to see if the loss function will decrease in a sort of "network process simulation".

Fake signals:

![alt text](fake_signals.png "fake signals")

Inputs of the loss function:

![alt text](loss_input.png "loss inputs")

Here we can observe the results of the loss function for a single frame:

![alt text](single_frame_loss.png "single_frame_loss")

As we can see with the smoothing process the loss value seems to decrease significantly, we can also notice that also the "fake sinusoidal signal" seems to perform better than the smoothed one.

## --> NOVEMBER 2021

## **Done**

## --> 07/03/2022

## **training performed**

- Try the normalization of the input and output given to the network as:

```{r test-python, engine='python'}
def normalize_audio(audio):
    max = np.max(audio)
    normalized_signal = np.array((audio / np.max(np.abs(audio))) * 32767, np.int16)           # Considering a 16 bit audio 
    return normalized_signal
    
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
```

## **Ideas for the next tasks**

### *Short period*

- Try to match the spectrum of the energy between the input and the target
- Understand how much the results change using the low pass filter
- Set the enviroment for the polimi machine --> ask Plosen

### *Long period*

- Try to train on the low frequencies only and on the high frequencies after

## --> 08/03/2022

## Results of the previous day

The results obtained from the last training are really bad, seems more a distortion simulation than an acoustic guitar :(

Here is the audio result:

<https://user-images.githubusercontent.com/55618574/157212559-fe78b37f-e68c-4ce1-98a5-311035dbe9c2.mp4>

And here there is the loss result (in log scale):

![GitHub Light](loss_07-03-2022.png "Loss")

## Done

- Training with normalization btw -1 and 1 of the input and target signal
- Start to update the Logbook with the past progress of the thesis

### *Short period*

- Finish to read and understand the paper "Effect_removal"
- Try to match the spectrum of the energy between the input and the target
- Understand how much the results change using the low pass filter
- Set the enviroment for the polimi machine --> ask Plosen
- Start writing the introduction

### *Long period*

- Try to train on the low frequencies only and on the high frequencies after
- Try to see if the paper "Effect_removal" could have some good implementation for our purpose
