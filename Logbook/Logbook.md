# ***LOGBOOK***

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

<https://github.com/EmanueleVoltolini/AI-Powered-Pickup/blob/code_update/Logbook/Distortion_07-03-2022.mov>
