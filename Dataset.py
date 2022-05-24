from turtle import fd
import numpy as np
import pandas as pd
import librosa
import numpy as np
import re 
import pre_processing as pp
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import join
import soundfile as sf

def extract_data(PATH):                  # extracts the data from the Dataset folder
    name = []
    ext = []
    length = []
    data = []
    dir_list = listdir(PATH)
    for filename in dir_list:
        name = re.search(r'(?<=)\w+', filename)
        audio , _ = librosa.load(join(PATH,filename),sr=44100)
        if (len(audio)%2==1):
            audio = np.delete(audio,len(audio)-1)
        first, second= np.split(audio,2)
        len_fisrt = len(first)
        len_second = len(second)
        ext = re.search(r'(?<=-)\w+', filename)
        data.append([name[0],ext[0],0, len_fisrt])
        data.append([name[0],ext[0],1, len_second])
    return data

def load_audio(X,y, DATASET_DIR):
    out_in = []
    out_tar = []
    for i in range(np.shape(X)[1]):
        audio_in, fs = librosa.load(DATASET_DIR + '/' + X.iat[i,0] + '-' + X.iat[i,1] + ".wav", sr=None)
        audio_tar, _ = librosa.load(DATASET_DIR + '/' + y.iat[i,0] + '-' + y.iat[i,1] + ".wav", sr=None)
        if(len(audio_in)%2 == 1):
            audio_in = np.delete(audio_in,len(audio_in)-1)
        if(len(audio_tar)%2 == 1):
            audio_tar = np.delete(audio_tar,len(audio_tar)-1)
        first_in, second_in = np.split(audio_in,2)
        first_tar, second_tar = np.split(audio_tar,2)
        if (X.iat[i,2] == 0 and y.iat[i,2] == X.iat[i,2]):
            out_in = np.concatenate((out_in,first_in))
            out_tar = np.concatenate((out_tar,first_tar))
        elif(X.iat[i,2] == 1 and y.iat[i,2] == X.iat[i,2]):
            out_in = np.concatenate((out_in,second_in))
            out_tar = np.concatenate((out_tar,second_tar))
    out = np.zeros([len(out_in),2])
    out[:,0] = pp.normalize_audio(out_in)
    out[:,1] = pp.normalize_audio(out_tar)
    return out

def load_data(CSV_DIR, DATASET_DIR):
    testfile = ["mixed_nc"]
    ext = ["input","target"]
    df = pd.read_csv(CSV_DIR)
    X = df.loc[df['ext'] == 'input']
    y = df.loc[df['ext'] == 'target']
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    X_train = df.loc[(df['ext'] == 'input') & (df['n_segm'] == 0) & (df['name']!= 'mixed_nc')]
    y_train = df.loc[(df['ext'] == 'target') & (df['n_segm'] == 0) & (df['name']!= 'mixed_nc')]
    X_val = df.loc[(df['ext'] == 'input') & (df['n_segm'] == 1) & (df['name']!= 'mixed_nc')]
    y_val = df.loc[(df['ext'] == 'target') & (df['n_segm'] == 1) & (df['name']!= 'mixed_nc')]
    X_test = df.loc[(df['name'] == 'mixed_nc') & (df['ext'] == 'input')]
    y_test = df.loc[(df['name'] == 'mixed_nc') & (df['ext'] == 'target')]
    traindata = load_audio(X_train,y_train,DATASET_DIR)
    validata = load_audio(X_val,y_val,DATASET_DIR)
    #testdata = load_audio(X_test,y_test,DATASET_DIR)
    testdata = pp.concatenate_audio(testfile, DATASET_DIR, ext)
    return traindata, validata, testdata


if __name__ == "__main__":
    DATA_DIR = "Dataset/"                # directory of the dataset
    CSV_DIR = "Dataset.csv"              # directory of the csv file
    header = ["name","ext","n_segm","length"]     # header of the csv file for the dataset
    #data_csv = extract_data(DATA_DIR)    # extract the useful data from the dataset folder 
    #f = open(CSV_DIR, 'w')               # write all the information of the dataset in the csv file
    #writer = csv.writer(f)
    #writer.writerow(header)
    #for i in range(np.shape(data_csv)[0]):
    #    writer.writerow(data_csv[i][:])
    #f.close()
    a, b, c = load_data(CSV_DIR, DATA_DIR)