# -*- coding: utf-8 -*-
"""

@reference: Deep Unfolded Robust PCA With Application to Clutter Suppression in Ultrasound. https://github.com/KrakenLeaf/CORONA

"""

import matplotlib
import sys

matplotlib.use('Agg')
sys.path.append('../')

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import pickle
import os
from DataSet_Unfolded import ImageDataset
from network.RPCAUNet import RPCAUNet, to_var
from classes.Dataset import Converter
from classes.Player import Player
from tqdm import tqdm


"""Settings"""
"""========================================================================="""
# Name and choice of training set
ProjectName = 'XCA'
prefix = 'RPCA-UNet'
# Load model
loadmodel = False
#mfile = './Results/.pkl'
mfile = './Para/Unfolded_train_3000.pkl'

"""Network Settings"""
params_net = {}
params_net['layers'] = 4
params_net['kernel'] = [(5, 1)] * 2 + [(3, 1)] * 2
params_net['coef_L'] = 0.4
params_net['coef_S'] = 1.8
params_net['CalInGPU'] = True  # whether to calculate in GPU
params_net['kernel'] = params_net['kernel'][0:params_net['layers']]
# Whether to plot predictions during training and frequency
plot = False
plotT = 1
if not plot:
    plt.ioff()
# seed
seed = 123
torch.manual_seed(seed)

# parameters for training
lr_list = [1e-4]
TrainInstances = 12000  # Size of training dataset
ValInstances = 2000    # Size of Validation dataset
BatchSize = 20
ValBatchSize = 20
ALPHA = 0.4  # Adjust the coefficient of foreground and background
num_epochs = 50
frame = 20

# Whether to calculate in GPU
CalInGPU = params_net['CalInGPU']
"""========================================================================="""

# Dataset, converter and player
data_dir = './Data/'
conter = Converter()
player = Player()
formshow = {'pred': 'concat', 'shape': (64, 64, 20)}
formlist = []
for i in range(6):
    formlist.append(formshow)
minloss = np.inf
# Logs
if not os.path.exists('Results'):
    os.mkdir('Results')
log = open('Results/%s_%s_Log_al%.2f_Tr%s_epoch%s_lr%.2e.txt' \
           % (ProjectName, prefix, ALPHA, TrainInstances, num_epochs, lr_list[0]), 'w')
print('Project Name: %s' % ProjectName)
print('params_net=\n%s\n' % str(params_net))
log.write('Project Name: %s\n' % ProjectName)
log.write('params_net=\n%s\n\n' % str(params_net))
# Loading data
print('Loading phase...')
print('----------------')
log.write('Loading phase...\n')
log.write('----------------\n')
shape_dset = (64, 64, 20)
# training dataset
train_dataset = ImageDataset(shape_dset, train=0, data_dir=data_dir, length=TrainInstances)
train_loader = data.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True)
# validation dataset
val_dataset = ImageDataset(shape_dset, train=1, data_dir=data_dir, length=ValInstances)
val_loader = data.DataLoader(val_dataset, batch_size=ValBatchSize, shuffle=True)
print('Finished loading.\n')
log.write('Finished loading.\n\n');

one = to_var(torch.ones([64,64,20]), CalInGPU)
# Training
for learning_rate in lr_list:
    # Construct network
    print('Configuring network...')
    log.write('Configuring network...\n')
    if not loadmodel:
        net = RPCAUNet(params_net)
    else:
        if mfile[-3:] == 'pkl':
            net = RPCAUNet(params_net)
            state_dict = torch.load(mfile, map_location='cpu')
            net.load_state_dict(state_dict)
        else:
            net = torch.load(mfile)

    if torch.cuda.is_available():
        net = net.cuda()

    # Loss and optimizer
    floss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Array for recording datas
    outputs_S = to_var(torch.zeros([64, 64, 20]), CalInGPU)
    outputs_L = to_var(torch.zeros([64, 64, 20]), CalInGPU)
    lossmean_vec = np.zeros((num_epochs,))
    lossmean_val_vec = np.zeros((num_epochs,))
    exp_vec_L = np.zeros((num_epochs, net.layers))
    exp_vec_S = np.zeros((num_epochs, net.layers))

    # Training
    print('Training the model over %d samples, with learning rate %.6f\n' \
          % (TrainInstances, learning_rate))
    log.write('Training the model over %d samples, with learning rate %.6f\n\n' \
              % (TrainInstances, learning_rate))

    noise = to_var(torch.rand([64, 64, 20]) / 1000, CalInGPU)
    # Run over each epoch
    for epoch in range(num_epochs):
        # print time
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        print('\n' + st)
        log.write('\n' + st + '\n')

        loss_val_mean = 0
        loss_mean = 0
        # Train
        print('Loading and calculating training batches...')
        log.write('Loading and calculating training batches...\n')
        starttime = time.time()
        n_iter = len(train_loader)
        with tqdm(total=n_iter, desc='Epoch: [%d/%d]' % (epoch+1, num_epochs), miniters=1) as t:
            for _, (L, S, D) in enumerate(train_loader):
                # set the gradients to zero at the beginning of each epoch
                optimizer.zero_grad()
                for ii in range(BatchSize):
                    # speed up@zrp
                    if torch.sum(S[ii]) == 0:
                        continue
                    inputs = to_var(D[ii], CalInGPU) + noise
                    targets_L = to_var(L[ii], CalInGPU) + noise
                    targets_S = to_var(S[ii], CalInGPU) + noise

                    # Forward + backward + loss
                    outputs_L, outputs_S= net(inputs)  # Forward
                    # Current loss
                    loss = ALPHA * floss(outputs_L, targets_L) + \
                        (1 - ALPHA) * floss(outputs_S, targets_S)
                    loss_mean += loss.item()

                    loss.backward()
                    # with torch.autograd.detect_anomaly():
                    #     loss.backward()

                optimizer.step()
                t.set_postfix_str("Batch Loss: %.4f" % (loss))
                # t.set_postfix_str("epoch Loss: %.4f" % loss_epoch.val)
                t.update()

        loss_mean = loss_mean / TrainInstances
        endtime = time.time()
        print('Training time is %f' % (endtime - starttime))
        log.write('Training time is %f\n' % (endtime - starttime))

        # Validation
        print('Loading and calculating validation batches...')
        log.write('Loading and calculating validation batches...\n')
        starttime = time.time()
        with torch.no_grad():
            for _, (Lv, Sv, Dv) in enumerate(val_loader):
                for jj in range(ValBatchSize):
                    inputsv = to_var(Dv[jj], CalInGPU)
                    targets_Lv = to_var(Lv[jj], CalInGPU)
                    targets_Sv = to_var(Sv[jj], CalInGPU)

                    outputs_Lv, outputs_Sv = net(inputsv)  # Forward
                    # Current loss
                    loss_val = ALPHA * floss(outputs_Lv, targets_Lv) + \
                               (1 - ALPHA) * floss(outputs_Sv, targets_Sv)
                    loss_val_mean += loss_val.item()
        loss_val_mean = loss_val_mean / ValInstances
        endtime = time.time()
        print('Test time is %f' % (endtime - starttime))
        log.write('Test time is %f\n' % (endtime - starttime))

        # Observe results
        if plot and ((epoch + 1) % plotT == 0 or epoch == 0):
            [xtr, ystr, pstr, xval, ysval, psval] = conter.torch2np([D[ii], S[ii], outputs_S,
                                                                     Dv[jj], Sv[jj], outputs_Sv],
                                                                    formlist)
            [yltr, pltr, ylval, plval] = conter.torch2np([L[ii], outputs_L,
                                                          Lv[jj], outputs_Lv],
                                                         formlist)
            player.plotmat([xtr[:, :, frame], ystr[:, :, frame], yltr[:, :, frame],
                            None, pstr[:, :, frame], pltr[:, :, frame],
                            xval[:, :, frame], ysval[:, :, frame], ylval[:, :, frame],
                            None, psval[:, :, frame], plval[:, :, frame]],
                           tit=['xtr', 'ystr', 'yltr', None, 'pstr', 'pltr',
                                'xval', 'ysval', 'ylval', None, 'psval', 'plval'],
                           supt='Epoch{%d/%d}' % (epoch + 1, num_epochs))
            plt.pause(0.1)

        lossmean_vec[epoch] = loss_mean
        lossmean_val_vec[epoch] = loss_val_mean
        exp_L, exp_S = net.getexp_LS()
        exp_vec_L[epoch, :] = exp_L
        exp_vec_S[epoch, :] = exp_S

        # Print loss
        if (epoch + 1) % 1 == 0:
            print('Epoch [%d/%d], Lossmean:%.5e, Validation lossmean:%.5e'
                  % (epoch + 1, num_epochs, loss_mean, loss_val_mean))
            log.write('Epoch [%d/%d], Lossmean:%.5e, Validation lossmean:%.5e\n'
                      % (epoch + 1, num_epochs, loss_mean, loss_val_mean))
            np.set_printoptions(precision=3)
            print('exp_L:', exp_L)
            print('exp_S:', exp_S)
            log.write('exp_L: ' + str(exp_L) + '\n')
            log.write('exp_S: ' + str(exp_S) + '\n')

            if loss.item() > 100:
                print('hitbadrut')
                log.write('hitbadrut\n')
                break

        # Save model in each epoch
        if True or loss_val_mean < minloss:
            print('saved at [epoch%d/%d]' \
                  % (epoch + 1, num_epochs))
            log.write('saved at [epoch%d/%d]\n' \
                      % (epoch + 1, num_epochs))
            torch.save(net.state_dict(),
                       'Results/%s_%s_Model_al%.2f_Tr%s_epoch%s_lr%.2e.pkl' \
                       % (ProjectName, prefix, ALPHA, TrainInstances,
                          num_epochs, learning_rate))
            minloss = min(loss_val_mean, minloss)

    """Save logs, prediction, loss figure, loss data, model and settings """
    # # Graphs
    # # Save the prediction figure
    # if not plot:
    #     [xtr, ystr, pstr, xval, ysval, psval] = conter.torch2np([D[ii], S[ii], outputs_S,
    #                                                              Dv[jj], Sv[jj], outputs_Sv],
    #                                                             formlist)
    #     [yltr, pltr, ylval, plval] = conter.torch2np([L[ii], outputs_L,
    #                                                   Lv[jj], outputs_Lv],
    #                                                  formlist)
    #     player.plotmat([xtr[:, :, frame], ystr[:, :, frame], yltr[:, :, frame],
    #                     None, pstr[:, :, frame], pltr[:, :, frame],
    #                     xval[:, :, frame], ysval[:, :, frame], ylval[:, :, frame],
    #                     None, psval[:, :, frame], plval[:, :, frame]],
    #                    tit=['xtr', 'ystr', 'yltr', None, 'pstr', 'pltr',
    #                         'xval', 'ysval', 'ylval', None, 'psval', 'plval'],
    #                    supt='Epoch{%d/%d}' % (epoch + 1, num_epochs), ion=False)
    #     plt.savefig('Results/%s_%s_Pred_al%.2f_Tr%s_epoch%s_lr%.2e.png' \
    #                 % (ProjectName, prefix, ALPHA, TrainInstances,
    #                    num_epochs, learning_rate))
    # # MSE
    # fig = plt.figure()
    # epochs_vec = np.arange(0, num_epochs, 1)
    # plt.semilogy(epochs_vec, lossmean_vec, '-*', label='loss')
    # plt.semilogy(epochs_vec, lossmean_val_vec, '-*', label='loss_val')
    # plt.xlabel('epoch')
    # plt.ylabel('Loss')
    # plt.ylim(ymin=0)
    # plt.title("MSE")
    # plt.legend()
    # plt.savefig("Results/%s_%s_LossPng_al%.2f_Tr%s_epoch%s_lr%.2e.png" \
    #             % (ProjectName, prefix, ALPHA, TrainInstances, num_epochs, learning_rate))
    # pickle.dump(fig, open("Results/%s_%s_LossFig_al%.2f_Tr%s_epoch%s_lr%.2e.fig.pickle" \
    #                       % (ProjectName, prefix, ALPHA, TrainInstances, num_epochs, learning_rate), 'wb'))
    # np.savez('Results/%s_%s_LossData_al%.2f_Tr%s_epoch%s_lr%.2e' \
    #          % (ProjectName, prefix, ALPHA, TrainInstances, num_epochs, learning_rate),
    #          lossmean_vec, lossmean_val_vec)
    #
    # # Lamb L
    # fig1 = plt.figure()
    # colormap = plt.cm.gist_ncar
    # plt.gca().set_prop_cycle(color=( \
    #     [colormap(i) for i in np.linspace(0, 0.9, 2 * params_net['layers'])]))
    # for i in range(net.layers):
    #     plt.plot(epochs_vec, exp_vec_L[:, i], label='%dth layer' % (i + 1))
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('Lambda L')
    # plt.title("Lambda L as function of epochs")
    # plt.savefig("Results/%s_%s_expLPng_al%.2f_Tr%s_epoch%s_lr%.2e.png" \
    #             % (ProjectName, prefix, ALPHA, TrainInstances, num_epochs, learning_rate))
    # pickle.dump(fig1, open("Results/%s_%s_expLFig_al%.2f_Tr%s_epoch%s_lr%.2e.fig.pickle" \
    #                        % (ProjectName, prefix, ALPHA, TrainInstances, num_epochs, learning_rate), 'wb'))
    #
    # # Lamb S
    # fig2 = plt.figure()
    # colormap = plt.cm.gist_ncar
    # plt.gca().set_prop_cycle(color=( \
    #     [colormap(i) for i in np.linspace(0, 0.9, 2 * params_net['layers'])]))
    # for i in range(net.layers):
    #     plt.plot(epochs_vec, exp_vec_S[:, i], label='%dth layer' % (i + 1))
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('Lambda S')
    # plt.title("Lambda S as function of epochs")
    # plt.savefig("Results/%s_%s_expSPng_al%.2f_Tr%s_epoch%s_lr%.2e.png" \
    #             % (ProjectName, prefix, ALPHA, TrainInstances, num_epochs, learning_rate))
    # pickle.dump(fig2, open("Results/%s_%s_expSFig_al%.2f_Tr%s_epoch%s_lr%.2e.fig.pickle" \
    #                        % (ProjectName, prefix, ALPHA, TrainInstances, num_epochs, learning_rate), 'wb'))
    #
    # # Save data of thresholding parameters for L, S
    # np.savez('Results/%s_%s_expLSData_al%.2f_Tr%s_epoch%s_lr%.2e' \
    #          % (ProjectName, prefix, ALPHA, TrainInstances, num_epochs, learning_rate)
    #          , exp_vec_L, exp_vec_S)
    #
    # # Save settings of training and network
    # params = {'ProjectName': ProjectName,
    #           'prefix': prefix,
    #           'model_loaded': mfile if loadmodel else None,
    #           'data_dir': data_dir,
    #           'shape': shape_dset,
    #           'lr_list': lr_list,
    #           'ALPHA': ALPHA,
    #           'TrainInstances': TrainInstances,
    #           'ValInstances': ValInstances,
    #           'BatchSize': BatchSize,
    #           'ValBatchSize': ValBatchSize,
    #           'num_epochs': num_epochs,
    #           'frame': frame}
    # file = open('Results/%s_%s_Settings_al%.2f_Tr%s_epoch%s_lr%.2e.txt' \
    #             % (ProjectName, prefix, ALPHA, TrainInstances, num_epochs, learning_rate), 'w')
    # file.write('Training Settings:\n\t')
    # for k, v in params.items():
    #     file.write(k + '=' + str(v) + '\n')
    #     file.write('\t')
    # # Save settings of Network
    # file.write('\nparams_net={')
    # numitem = 1
    # for k, v in params_net.items():
    #     file.write("'" + k + "'" + ':' + str(v))
    #     if numitem < len(params_net):
    #         file.write(',\n\t\t\t  ')
    #     else:
    #         file.write('}\n')
    #     numitem += 1
    # file.close()
    #
    # # Print min loss
    # print('\nmin Loss=%.4e' % np.min(lossmean_val_vec))
    # log.write('\nmin Loss=%.4e\n' % np.min(lossmean_val_vec))
    # log.close()