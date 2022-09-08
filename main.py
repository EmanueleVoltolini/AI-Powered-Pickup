# Import all the necessary libraries
import torch
import os
import time
import Network as net
import pre_processing as pp
import general as gen
import Dataset as data
from torch.utils.tensorboard import SummaryWriter
from numpy import double, pi, log10
from scipy.io.wavfile import write

if __name__ == "__main__":
    start_time = time.time()

    ############################################################################################################
    ############################################# PARAMETERS ###################################################
    ############################################################################################################
    DATASET_DIR = 'Dataset.csv'
    #DATASET_DIR = 'Dataset_Taylor.csv'
    AUDIO_DIR = "Dataset"
    #AUDIO_DIR = "Dataset_Taylor"
    CSV_DIR = "Dataset.csv"
    #CSV_DIR = "Dataset_Taylor.csv"
    save_path = "Results"
    PATH = 'Model'
    EPOCHS = 1
    LEARNING_RATE = 0.01
    up_freq = 2048
    n_segments = 40
    ext = ["input", "target"]  # extension of the audio dataset
    fs = 44100
    batch_size = 40
    model_name = "RNN"
    n_shuffle = 7  # number of segment for each shuffled group
    segment_len = int(fs*7)

    # Filter requirements
    order = 6  # sample rate, Hz
    cutoff = 2000  # desired cutoff frequency of the filter, Hz
    high_cut = 10000
    overlap = 0.75
    validation_f = 5  # validation frequency in number of epochs
    validation_p = 200  # validation patient in number of epochs

    ############################################################################################################
    #################################### SPLITTING AND PREPROCESSING ###########################################
    ############################################################################################################

    # concatenate the audio to obtain a single audio containing all the data
    traindata, validata, testdata = data.load_data(CSV_DIR, AUDIO_DIR)
    
    # Smoothing the input signal given to the network to have a better comparison with the target 
    traindata[:, 0] = pp.smooth_signal(traindata[:, 0])
    validata[:, 0] = pp.smooth_signal(validata[:, 0])
    testdata[:, 0] = pp.smooth_signal(testdata[:, 0])

    # Apply different filters to the signal input signal 
    traindata[:,0] = pp.butter_lowpass_filter(traindata[:, 0], cutoff, fs, order)
    #traindata[:, 0] = pp.butter_bandpass_filter(traindata[:, 0], cutoff, high_cut, fs, order)
    #traindata[:, 1] = pp.butter_bandpass_filter(traindata[:, 1], cutoff, high_cut, fs, order)

    # Splitting the audio in overlapping windowed segments that match the input dimension of the network
    train_in = pp.split_audio_overlap(traindata[:, 0], segment_len, overlap)
    train_tar = pp.split_audio_overlap(traindata[:, 1], segment_len, overlap)

    val_in = pp.split_audio_overlap(validata[:, 0], segment_len, overlap)
    val_tar = pp.split_audio_overlap(validata[:, 1], segment_len, overlap)

    test_in = pp.split_audio_overlap(testdata[:, 0], segment_len, overlap)
    test_tar = pp.split_audio_overlap(testdata[:, 1], segment_len, overlap)

    ############################################################################################################
    ########################################## NETWORKS PARAMETERS #############################################
    ############################################################################################################

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

    # Defining the used optimizer
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Defining the loss function
    loss_functions = net.Loss()
    train_track = net.TrainTrack()

    # Defining the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.7, patience=3, verbose=True)

    # Initialization of other parameters 
    init_time = time.time() - start_time + train_track['total_time'] * 3600
    patience_counter = 0
    writer = SummaryWriter(os.path.join('runs2', model_name))

    ############################################################################################################
    ############################################# TRAINING #####################################################
    ############################################################################################################

    min_loss = 1e10

    for epoch in range(EPOCHS):
        ep_st_time = time.time()
        if epoch == 500:
            optimiser.param_groups[0]['lr'] = LEARNING_RATE * 0.8
        if epoch == 700:
            optimiser.param_groups[0]['lr'] = LEARNING_RATE * 0.1
        #    optimiser.param_groups[0]['weight_decay'] = 1e-6
        # Train one epoch
        epoch_loss = network.train_one_epoch(train_in, train_tar, up_freq, 1000, batch_size, optimiser, loss_functions,
                                             n_shuffle)

        # Run validation
        if epoch % validation_f == 0:
            val_ep_st_time = time.time()
            val_output, val_loss = network.process_data(val_in,
                                                        val_tar, loss_functions, 10)
            scheduler.step(val_loss)
            if val_loss < train_track['best_val_loss']:
                patience_counter = 0
                torch.save(network.state_dict(), os.path.join(save_path, 'model_best'))
                write(os.path.join(save_path, "best_val_out.wav"),
                      fs, val_output.cpu().numpy()[:, 0, 0])
            else:
                patience_counter += 1
            train_track.val_epoch_update(val_loss.item(), val_ep_st_time, time.time())
            writer.add_scalar('Loss/val', train_track['validation_losses'][-1], epoch)
        if epoch % 10 == 0:
            print('current learning rate: ' + str(optimiser.param_groups[0]['lr']))
        train_track.train_epoch_update(epoch_loss.item(), ep_st_time, time.time(), init_time, epoch)
        writer.add_scalar('Loss/train', train_track['training_losses'][-1], epoch)
        writer.add_scalar('LR/current', optimiser.param_groups[0]['lr'], epoch)

        if patience_counter > validation_p:
            print('validation patience limit reached at epoch ' + str(epoch))
            break
    lossESR = net.ESRLoss()
    test_output, test_loss = network.process_data(test_in,
                                                  test_tar, loss_functions, 10)
    test_loss_ESR = lossESR(test_output, test_tar)
    write(os.path.join(save_path, "test_out_final.wav"), fs, test_output.cpu().numpy()[:, 0, 0])
    writer.add_scalar('Loss/test_loss', test_loss.item(), 1)
    writer.add_scalar('Loss/test_lossESR', test_loss_ESR.item(), 1)
    train_track['test_loss_final'] = test_loss.item()
    train_track['test_lossESR_final'] = test_loss_ESR.item()

    best_val_net = torch.load(os.path.join(save_path, 'model_best'))
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
    gen.json_save(train_track, 'training_stats', save_path)
