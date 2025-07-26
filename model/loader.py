import os
import numpy as np
import matplotlib.pyplot as plt
import random
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from datetime import date

import torch
from torchvision import transforms, datasets
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

from pylab import mpl

import warnings

from einops import rearrange
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import math

from matplotlib.collections import LineCollection

from matplotlib.colors import ListedColormap, BoundaryNorm

from model.sequence_cmt import Epoch_Cross_Transformer
from model.model_blocks import PositionalEncoding, Window_Embedding, Intra_modal_atten, Cross_modal_atten, Feed_forward



class SleepEDF_Seq_MultiChan_Dataset_Inference(Dataset):
    def __init__(self, eeg_file, eog_file, device, mean_eeg_l=None, sd_eeg_l=None,
                 mean_eog_l=None, sd_eog_l=None, mean_eeg2_l=None, sd_eeg2_l=None, transform=None,
                 target_transform=None, sub_wise_norm=False, num_seq=5):
        """

        """
        # Get the data

        self.eeg = eeg_file
        self.eog = eog_file
        # self.labels = label_file

        # self.labels = torch.from_numpy(self.labels)

        # bin_labels = np.bincount(self.labels)
        # print(f"Labels count: {bin_labels}")
        print(f"Shape of EEG : {self.eeg.shape} , EOG : {self.eog.shape}")  # , EMG: {self.eeg2.shape}")
        # print(f"Shape of Labels : {self.labels.shape}")

        if sub_wise_norm == True:
            print(f"Reading Subject wise mean and sd")

            self.mean_eeg = mean_eeg_l
            self.sd_eeg = sd_eeg_l
            self.mean_eog = mean_eog_l
            self.sd_eog = sd_eog_l

        self.sub_wise_norm = sub_wise_norm
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.num_seq = num_seq

    def __len__(self):
        return self.eeg.shape[0] - self.num_seq

    def __getitem__(self, idx):
        eeg_data = self.eeg[idx:idx + self.num_seq].squeeze()
        eog_data = self.eog[idx:idx + self.num_seq].squeeze()
        # label = self.labels[idx:idx + self.num_seq, ]

        if self.sub_wise_norm == True:
            eeg_data = (eeg_data - self.mean_eeg[idx]) / self.sd_eeg[idx]
            eog_data = (eog_data - self.mean_eog[idx]) / self.sd_eog[idx]
        elif self.mean and self.sd:
            eeg_data = (eeg_data - self.mean[0]) / self.sd[0]
            eog_data = (eog_data - self.mean[1]) / self.sd[1]
        if self.transform:
            eeg_data = self.transform(eeg_data)
            eog_data = self.transform(eog_data)
        # if self.target_transform:
        # label = self.target_transform(label)
        return eeg_data, eog_data

class Seq_Cross_Transformer_Network(nn.Module):
    def __init__(self, d_model=128, dim_feedforward=512, window_size=25):  # filt_ch = 4
        super(Seq_Cross_Transformer_Network, self).__init__()

        self.epoch_1 = Epoch_Cross_Transformer(d_model=d_model, dim_feedforward=dim_feedforward,
                                               window_size=window_size)
        self.epoch_2 = Epoch_Cross_Transformer(d_model=d_model, dim_feedforward=dim_feedforward,
                                               window_size=window_size)
        self.epoch_3 = Epoch_Cross_Transformer(d_model=d_model, dim_feedforward=dim_feedforward,
                                               window_size=window_size)
        self.epoch_4 = Epoch_Cross_Transformer(d_model=d_model, dim_feedforward=dim_feedforward,
                                               window_size=window_size)
        self.epoch_5 = Epoch_Cross_Transformer(d_model=d_model, dim_feedforward=dim_feedforward,
                                               window_size=window_size)
        #
        self.epoch_6 = Epoch_Cross_Transformer(d_model=d_model, dim_feedforward=dim_feedforward,
                                               window_size=window_size)
        self.epoch_7 = Epoch_Cross_Transformer(d_model=d_model, dim_feedforward=dim_feedforward,
                                               window_size=window_size)
        self.epoch_8 = Epoch_Cross_Transformer(d_model=d_model, dim_feedforward=dim_feedforward,
                                               window_size=window_size)
        self.epoch_9 = Epoch_Cross_Transformer(d_model=d_model, dim_feedforward=dim_feedforward,
                                               window_size=window_size)
        self.epoch_10 = Epoch_Cross_Transformer(d_model=d_model, dim_feedforward=dim_feedforward,
                                                window_size=window_size)
        #
        self.epoch_11 = Epoch_Cross_Transformer(d_model=d_model, dim_feedforward=dim_feedforward,
                                                window_size=window_size)
        self.epoch_12 = Epoch_Cross_Transformer(d_model=d_model, dim_feedforward=dim_feedforward,
                                                window_size=window_size)
        self.epoch_13 = Epoch_Cross_Transformer(d_model=d_model, dim_feedforward=dim_feedforward,
                                                window_size=window_size)
        self.epoch_14 = Epoch_Cross_Transformer(d_model=d_model, dim_feedforward=dim_feedforward,
                                                window_size=window_size)
        self.epoch_15 = Epoch_Cross_Transformer(d_model=d_model, dim_feedforward=dim_feedforward,
                                                window_size=window_size)
        #
        # self.epoch_16 = Epoch_Cross_Transformer(d_model = d_model, dim_feedforward=dim_feedforward,
        #                                         window_size = window_size)
        # self.epoch_17 = Epoch_Cross_Transformer(d_model = d_model, dim_feedforward=dim_feedforward,
        #                                         window_size = window_size)
        # self.epoch_18 = Epoch_Cross_Transformer(d_model = d_model, dim_feedforward=dim_feedforward,
        #                                         window_size = window_size)
        # self.epoch_19 = Epoch_Cross_Transformer(d_model = d_model, dim_feedforward=dim_feedforward,
        #                                         window_size = window_size)
        # self.epoch_20 = Epoch_Cross_Transformer(d_model = d_model, dim_feedforward=dim_feedforward,
        #                                         window_size = window_size)
        # #
        #         # self.epoch_21 = Epoch_Cross_Transformer(d_model = d_model, dim_feedforward=dim_feedforward,
        #                                                 window_size = window_size)

        self.seq_atten = Intra_modal_atten(d_model=d_model, nhead=8, dropout=0.1, window_size=window_size, First=False)

        self.ff_net = Feed_forward(d_model=d_model, dropout=0.1, dim_feedforward=dim_feedforward)

        self.mlp_1 = nn.Sequential(nn.Flatten(), nn.Linear(d_model, 5))  ##################
        self.mlp_2 = nn.Sequential(nn.Flatten(), nn.Linear(d_model, 5))
        self.mlp_3 = nn.Sequential(nn.Flatten(), nn.Linear(d_model, 5))
        self.mlp_4 = nn.Sequential(nn.Flatten(), nn.Linear(d_model, 5))
        self.mlp_5 = nn.Sequential(nn.Flatten(), nn.Linear(d_model, 5))
        #
        self.mlp_6 = nn.Sequential(nn.Flatten(), nn.Linear(d_model, 5))  ##################
        self.mlp_7 = nn.Sequential(nn.Flatten(), nn.Linear(d_model, 5))
        self.mlp_8 = nn.Sequential(nn.Flatten(), nn.Linear(d_model, 5))
        self.mlp_9 = nn.Sequential(nn.Flatten(), nn.Linear(d_model, 5))
        self.mlp_10 = nn.Sequential(nn.Flatten(), nn.Linear(d_model, 5))
        #
        self.mlp_11 = nn.Sequential(nn.Flatten(), nn.Linear(d_model, 5))  ##################
        self.mlp_12 = nn.Sequential(nn.Flatten(), nn.Linear(d_model, 5))
        self.mlp_13 = nn.Sequential(nn.Flatten(), nn.Linear(d_model, 5))
        self.mlp_14 = nn.Sequential(nn.Flatten(), nn.Linear(d_model, 5))
        self.mlp_15 = nn.Sequential(nn.Flatten(), nn.Linear(d_model, 5))
        #
        # self.mlp_16    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))  ##################
        # self.mlp_17    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        # self.mlp_18    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        # self.mlp_19    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        # self.mlp_20    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        # self.mlp_21    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))

    def forward(self, eeg: Tensor, eog: Tensor, num_seg=5):
        # eeg_epoch = eeg[:,:,0,:]
        # eog_epoch = eog[:,:,0,:]
        # for ep in range(1,num_seg):
        #     eeg_epoch = torch.cat((eeg_epoch,eeg[:,:,ep,:]),dim=-1)
        #     eog_epoch = torch.cat((eog_epoch,eog[:,:,ep,:]),dim=-1)

        # print(eeg_epoch.shape,eog_epoch.shape)
        epoch_1, feat_1 = self.epoch_1(eeg[:, :, 0, :], eog[:, :, 0, :])  # [0]
        epoch_2, feat_2 = self.epoch_2(eeg[:, :, 1, :], eog[:, :, 1, :])  # [0]
        epoch_3, feat_3 = self.epoch_3(eeg[:, :, 2, :], eog[:, :, 2, :])  # [0]
        epoch_4, feat_4 = self.epoch_4(eeg[:, :, 3, :], eog[:, :, 3, :])  # [0]
        epoch_5, feat_5 = self.epoch_5(eeg[:, :, 4, :], eog[:, :, 4, :])  # [0]
        # print(epoch_1.shape,epoch_5.shape)
        epoch_6, feat_6 = self.epoch_6(eeg[:, :, 5, :], eog[:, :, 5, :])  # [0]
        epoch_7, feat_7 = self.epoch_7(eeg[:, :, 6, :], eog[:, :, 6, :])  # [0]
        epoch_8, feat_8 = self.epoch_8(eeg[:, :, 7, :], eog[:, :, 7, :])  # [0]
        epoch_9, feat_9 = self.epoch_9(eeg[:, :, 8, :], eog[:, :, 8, :])  # [0]
        epoch_10, feat_10 = self.epoch_10(eeg[:, :, 9, :], eog[:, :, 9, :])  # [0]
        # print(epoch_1.shape,epoch_5.shape)
        epoch_11, feat_11 = self.epoch_11(eeg[:, :, 10, :], eog[:, :, 10, :])  # [0]
        epoch_12, feat_12 = self.epoch_12(eeg[:, :, 11, :], eog[:, :, 11, :])  # [0]
        epoch_13, feat_13 = self.epoch_13(eeg[:, :, 12, :], eog[:, :, 12, :])  # [0]
        epoch_14, feat_14 = self.epoch_14(eeg[:, :, 13, :], eog[:, :, 13, :])  # [0]
        epoch_15, feat_15 = self.epoch_15(eeg[:, :, 14, :], eog[:, :, 14, :])  # [0]
        # print(epoch_1.shape,epoch_5.shape)
        # epoch_16 = self.epoch_16(eeg[:,:,15,:],eog[:,:,15,:])[0]
        # epoch_17 = self.epoch_17(eeg[:,:,16,:],eog[:,:,16,:])[0]
        # epoch_18 = self.epoch_18(eeg[:,:,17,:],eog[:,:,17,:])[0]
        # epoch_19 = self.epoch_19(eeg[:,:,18,:],eog[:,:,18,:])[0]
        # epoch_20 = self.epoch_20(eeg[:,:,19,:],eog[:,:,19,:])[0]
        # # print(epoch_1.shape,epoch_5.shape)
        # epoch_21 = self.epoch_21(eeg[:,:,20,:],eog[:,:,20,:])[0]

        # seq =  torch.cat([epoch_1, epoch_2,epoch_3,epoch_4,epoch_5], dim=1)
        seq = torch.cat([epoch_1, epoch_2, epoch_3, epoch_4, epoch_5,
                         epoch_6, epoch_7, epoch_8, epoch_9, epoch_10,
                         epoch_11, epoch_12, epoch_13, epoch_14, epoch_15], dim=1)
        # epoch_16, epoch_17,epoch_18,epoch_19,epoch_20,epoch_21], dim=1)
        seq = self.seq_atten(seq)
        # print(seq.shape)
        seq = self.ff_net(seq)
        # print(seq.shape)
        out_1 = self.mlp_1(seq[:, 0, :])
        out_2 = self.mlp_2(seq[:, 1, :])
        out_3 = self.mlp_3(seq[:, 2, :])
        out_4 = self.mlp_4(seq[:, 3, :])
        out_5 = self.mlp_5(seq[:, 4, :])
        #
        out_6 = self.mlp_6(seq[:, 5, :])
        out_7 = self.mlp_7(seq[:, 6, :])
        out_8 = self.mlp_8(seq[:, 7, :])
        out_9 = self.mlp_9(seq[:, 8, :])
        out_10 = self.mlp_10(seq[:, 9, :])
        #
        out_11 = self.mlp_11(seq[:, 10, :])
        out_12 = self.mlp_12(seq[:, 11, :])
        out_13 = self.mlp_13(seq[:, 12, :])
        out_14 = self.mlp_14(seq[:, 13, :])
        out_15 = self.mlp_15(seq[:, 14, :])
        #
        #         out_16 = self.mlp_16(seq[:,15,:])
        #         out_17 = self.mlp_17(seq[:,16,:])
        #         out_18 = self.mlp_18(seq[:,17,:])
        #         out_19 = self.mlp_19(seq[:,18,:])
        #         out_20 = self.mlp_20(seq[:,19,:])

        #         out_21 = self.mlp_21(seq[:,20,:])
        feat_list = [feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, feat_7, feat_8, feat_9, feat_10, feat_11, feat_12,
                     feat_13, feat_14, feat_15, seq]
        # print(out_1.shape)
        return [out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14,
                out_15], feat_list  # ,out_16,out_17,out_18,out_19,out_20,out_21]