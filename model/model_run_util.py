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

from model.sequence_cmt import Seq_Cross_Transformer_Network  # as Seq_Cross_Transformer_Network
from model.sequence_cmt import Epoch_Cross_Transformer
from model.model_blocks import PositionalEncoding, Window_Embedding, Intra_modal_atten, Cross_modal_atten, Feed_forward

import json

from data_preparations.new_single_epoch_Sleep_EDF_153 import Sleep_EDF_SC_signal_extract_WITHOUT_HY

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

def plot_interpret(i, x, y, dydx, fig, axs, axs_no, signal_type="EEG"):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # plt.figure(figsize = (30,5))
    # plt.figure(figsize=(25,5))
    # plt.plot(x,dydx)
    # plt.title(f"Attention Map for Class {label}  {signal_type} ")
    # plt.xlim(x.min(),x.max())
    # plt.colorbar()

    # fig, axs = plt.subplots(2, 1, sharex=True, sharey=True,figsize = (30,10))

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())

    lc = LineCollection(segments, cmap='Reds', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(15)
    line = axs[axs_no[0]][axs_no[1]].add_collection(lc)
    # fig.colorbar(line, ax=axs[axs_no[0]][axs_no[1]])
    # fig.colorbar(line, ax=axs[1])
    # axs[axs_no[0]][axs_no[1]].set_xlabel(f"{signal_type}",fontsize = 100,labelpad = 20)
    axs[axs_no[0]][axs_no[1]].set_title(f'Epoch {i + 1} {signal_type}', fontsize=100)
    # axs[i].set_xlabel('Signal',fontsize = 100)
    # axs[axs_no[0]][axs_no[1]].axis('off')
    # Hide X and Y axes label marks
    axs[axs_no[0]][axs_no[1]].xaxis.set_tick_params(labelbottom=False)
    axs[axs_no[0]][axs_no[1]].yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    axs[axs_no[0]][axs_no[1]].set_xticks([])
    axs[axs_no[0]][axs_no[1]].set_yticks([])
    axs[axs_no[0]][axs_no[1]].set_xlim(x.min(), x.max())
    axs[axs_no[0]][axs_no[1]].set_ylim(y.min() - 0.2, y.max() + 0.2)

def atten_interpret(q, k):
    atten_weights = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1)
    return atten_weights


def model_run(Psg_file, out_dir, model_config, start_time=0):
    """数据预处理"""
    eeg_raw_data, eeg_sub_len, eeg_mean, eeg_std = (
        Sleep_EDF_SC_signal_extract_WITHOUT_HY(Psg_file, channel="eeg1", filter=True, freq=[0.2, 40])
    )
    eog_raw_data, _, eog_mean, eog_std = (
        Sleep_EDF_SC_signal_extract_WITHOUT_HY(Psg_file, channel="eog", filter=True, freq=[0.2, 40])
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_seq = model_config["use"]["num_seq"]

    """生成数据加载器"""
    infer_dataset = SleepEDF_Seq_MultiChan_Dataset_Inference(eeg_file=eeg_raw_data,
                                                             eog_file=eog_raw_data,
                                                             # label_file=eeg_labels,
                                                             device=device, mean_eeg_l=eeg_mean, sd_eeg_l=eeg_std,
                                                             mean_eog_l=eog_mean, sd_eog_l=eog_std,
                                                             sub_wise_norm=True, num_seq=num_seq,
                                                             transform=transforms.Compose([
                                                                 transforms.ToTensor()
                                                             ]))

    infer_data_loader = data.DataLoader(infer_dataset, batch_size=1, shuffle=False)  # 16
    eeg_data, eog_data = next(iter(infer_data_loader))

    # print(f"EEG batch shape: {eeg_data.size()}")
    # print(f"EOG batch shape: {eog_data.size()}")
    # print(f"EMG batch shape: {eeg2_data.size()}")
    # print(f"Labels batch shape: {label.size()}")

    ##第一张图片: eeg_sample
    eeg_data_temp = eeg_data[0].squeeze()  # (0)
    eog_data_temp = eog_data[0].squeeze()  # (0)

    # print(eeg_data_temp.shape)

    t = np.arange(0, 30, 1 / 100)
    plt.figure(figsize=(10, 5))
    plt.plot(eeg_data_temp[0].squeeze())
    plt.plot(eog_data_temp[0].squeeze() + 5)
    plt.title(f"EEG & EOG表格")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_path = os.path.join(out_dir, "eeg_sample.jpg")
    plt.savefig(save_path, dpi=300)

    """ 部署模型"""
    test_model = torch.load(model_config["use"]["model_path"], map_location=device, weights_only=False)
    test_model.eval()
    # print(sum(p.numel() for p in test_model.parameters() if p.requires_grad))

    """运行模型得到结果"""
    warnings.filterwarnings("ignore")

    # feat_main = []
    pred_val_main = torch.zeros((len(infer_data_loader) + num_seq, 1, 5))  # data, output,seq pred,
    # labels_val_main = torch.zeros((len(infer_data_loader) + num_seq, 1))  # data, output,seq pred,
    m = torch.nn.Softmax()
    with torch.no_grad():  # 确保在接下来的代码块中不会计算梯度
        test_model.eval()  # 将模型设置为评估模式，这对于推断是必要的
        for batch_val_idx, data_val in enumerate(infer_data_loader):  # 遍历数据加载器中的每个批次
            # if batch_val_idx % 1 == 0:
                # print("predicting", batch_val_idx)
            val_eeg, val_eog = data_val  # 从批次数据中解包EEG、EOG信号和标签
            pred, _ = test_model(val_eeg.float().to(device), val_eog.float().to(device))  # 使用模型进行预测，忽略返回的第二个值

            # print("#########")
            # print("Start")
            # print(pred)
            # print("End")
            # print("#########")

            # feat_main.append(feat_list)  # 这行代码被注释掉了，它看起来像是用来存储特征的
            for ep in range(num_seq):  # 遍历每个序列
                pred_val_main[batch_val_idx + ep] += m(pred[ep]).cpu()  # 将预测结果累加到pred_val_main数组中，m可能是一个映射函数
    # print(pred_val_main[0],pred_val_main[1000])
    pred_val_main = (pred_val_main / num_seq).squeeze()  # 计算预测的平均值

    """获取各个睡眠阶段在整个睡眠时间的占比"""
    # 1. 获取每个时间步的预测类别索引
    pred_labels = torch.argmax(pred_val_main, dim=1)
    # 2. 统计各睡眠阶段的数量
    stage_counts = torch.bincount(pred_labels, minlength=5)
    # 3. 计算总时间步数
    total_steps = pred_labels.size(0)
    # 4. 创建睡眠阶段名称映射
    stage_mapping = {
        0: "w",  # Wake
        1: "n1",  # NREM Stage 1
        2: "n2",  # NREM Stage 2
        3: "n3",  # NREM Stage 3
        4: "rem"  # REM Sleep
    }
    # 5. 计算各阶段占比百分比（四舍五入到整数）
    result_dict = {}
    for i in range(5):
        stage_name = stage_mapping[i]
        percentage = round(stage_counts[i].item() / total_steps * 100)
        result_dict[stage_name] = int(percentage)
    # 6. 确保百分比总和为100%（调整可能的四舍五入误差）
    total_percentage = sum(result_dict.values())
    if total_percentage != 100:
        # 找到最大占比的阶段进行调整
        max_stage = max(result_dict, key=result_dict.get)
        result_dict[max_stage] += 100 - total_percentage
    """获取预测pred"""
    # 将索引映射为阶段名称
    pred_stages = [stage_mapping[idx.item()] for idx in pred_labels]

    """这上面要生成一个预测结果的文件，供提取睡眠分期结构使用(?存疑，不知道提取睡眠结构特征的函数要的是怎么样的文件)"""

    """得到可解释性图像"""
    batch_size = len(infer_data_loader)
    infer_data_loader = data.DataLoader(infer_dataset, batch_size=batch_size, shuffle=False)  # 16

    t = start_time

    eeg_data, eog_data = next(iter(infer_data_loader))

    l = eeg_data.shape[0] // num_seq

    # for i in range(l):
    #     pred, feat_list = test_model(eeg_data[i*num_seq].unsqueeze(0).float().to(device), eog_data[i*num_seq].unsqueeze(0).float().to(device))
    #     pred = np.array([i.argmax(-1).item() for i in pred])
    #     for j in range(num_seq):
    #         new_pred.append(pred[j])

    pred, feat_list = test_model(eeg_data[t].unsqueeze(0).float().to(device),
                                 eog_data[t].unsqueeze(0).float().to(device))
    pred = np.array([i.argmax(-1).item() for i in pred])

    label_dict = ['Wake', 'N1', 'N2', 'N3', 'REM']
    pred_list = [label_dict[i] for i in pred]
    # print("pred_list",pred_list)
    # print("new_pred_list",new_pred)

    # data_new = [
    #     {'name': 'N1期', 'value': 0},
    #     {'name': 'N2期', 'value': 0},
    #     {'name': 'N3期', 'value': 0},
    #     {'name': 'REM期', 'value': 0}
    # ]
    #
    # for i in new_pred:
    #     if i == 1:
    #         data_new[0]['value'] += 0.5
    #     elif i == 2:
    #         data_new[1]['value'] += 0.5
    #     elif i == 3:
    #         data_new[2]['value'] += 0.5
    #     elif i == 4:
    #         data_new[3]['value'] += 0.5

    # 定义文件夹和文件名
    # directory = "sleep_proportion"  # 存储数据的文件夹名称
    # file_name = "sleep_data.json"  # 存储数据的文件名称
    # file_path = os.path.join(directory, file_name)  # 拼接完整的文件路径
    #
    # # 如果文件夹不存在，则创建文件夹
    # if not os.path.exists(directory):
    #     os.makedirs(directory)  # 创建文件夹
    #
    # # 将data_new数据保存到JSON文件中
    # with open(file_path, 'w') as file:  # 打开文件用于写入
    #     json.dump(data_new, file)  # 将data_new转换为JSON格式并写入文件

    #########################################################################################################################################
    # ## 画图 ##
    # ###### Interpreting inter-epoch relationships  ##########
    # plt.rcParams['axes.linewidth'] = 2
    seq_features = feat_list[-1]  ##extracting learned inter-epoch features
    # # seq_atten = atten_interpret(seq_features.squeeze(),seq_features.squeeze()).squeeze().detach().cpu().numpy()
    # # print(seq_atten.shape)
    # # plt.figure()
    # # plt.imshow(seq_atten)
    #
    # fig, axs = plt.subplots(5, 1, figsize=(1 * 5, 15 * 8))
    seq_atten_list = []
    for i in range(num_seq):
        seq_atten = atten_interpret(seq_features.squeeze()[i].unsqueeze(0),
                                    seq_features.squeeze()).squeeze().detach().cpu().numpy()
        #
        #     rgba_colors = np.zeros((num_seq, 4))
        #     rgba_colors[:, 0] = 0  # value of red intensity divided by 256
        #     rgba_colors[i, 0] = 0.4  # value of red intensity divided by 256
        #     rgba_colors[:, 1] = 0  # value of green intensity divided by 256
        #     rgba_colors[:, 2] = 0.4  # value of blue intensity divided by 256
        #     rgba_colors[i, 2] = 0
        seq_atten = seq_atten / seq_atten.max()
        #
        seq_atten_list.append(seq_atten)  #
    #     rgba_colors[:, -1] = seq_atten
    #     ###############################################################################
    #     # axs[i].bar(np.arange(1, 16), seq_atten / seq_atten.max(),  # color ='blue',
    #     #            color=rgba_colors, align='center', width=0.8)
    #     axs[i].bar(np.arange(1, 6), seq_atten / seq_atten.max(),  # color ='blue',
    #                color=rgba_colors, align='center', width=0.8)
    #     ###############################################################################
    #     # axs[i//5][i%5].set_title('')
    #     axs[i].tick_params(axis='x', labelsize=30)  # ,which = 'both')
    #     axs[i].tick_params(axis='y', labelsize=30)
    #     axs[i].set_xlabel('Epochs', fontsize=30)
    #     yticks = axs[i].yaxis.get_major_ticks()
    #     yticks[0].label1.set_visible(False)
    #
    # save_dir = "static/picture"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # save_path = os.path.join(save_dir, "picture_2.jpg")
    # plt.savefig(save_path, dpi=300)
    #
    """第二张图：模态间关系图"""
    fig, axs = plt.subplots(15, 1, figsize=(1 * 5, 15 * 10))
    fig, axs = plt.subplots(5, 1, figsize=(1 * 5, 15 * 10))


    cross_atten_list = []  #
    # from matplotlib.font_manager import FontProperties
    # my_font = FontProperties(fname='env/simhei.ttf')
    #
    for i in range(num_seq):
        cross_features = feat_list[i][-1]  ##extracting learned cross-modal features
        cross_atten = atten_interpret(cross_features.squeeze()[0].unsqueeze(0),
                                      cross_features.squeeze()[1:]).squeeze().detach().cpu().numpy()
        cross_atten_list.append(cross_atten)  #

        rgba_colors = np.zeros((2, 4))
        rgba_colors[:, 0] = 0.4  # value of red intensity divided by 256
        rgba_colors[:, 1] = 0  # value of green intensity divided by 256
        rgba_colors[:, 2] = 0  # value of blue intensity divided by 256
        rgba_colors[:, -1] = cross_atten + 0.1
        axs[i].bar(['EEG', 'EOG'], cross_atten,  # color ='red',
                   color=rgba_colors, align='center', width=0.9)
        axs[i].tick_params(axis='x', labelsize=30)  # ,which = 'both')
        axs[i].tick_params(axis='y', labelsize=30)
        axs[i].set_ylim(0, 1.02)
        # axs[i].set_xlabel('注意力占比', fontsize=30, fontproperties=my_font)
        axs[i].set_xlabel('注意力占比', fontsize=30)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_path = os.path.join(out_dir, "relationships.jpg")
    plt.savefig(save_path, dpi=300)
    # # plt.savefig(f'/content/cross_modal_sub_{subject_no}_day_{days}_t_{t}_part_1.pdf',dpi = 300)
    #
    # ###### Interpreting intra-modal relationships  ##########
    #     ###### 解释同模态内部关系  ##########
    # plt.rcParams['axes.linewidth'] = 20
    #
    # ###############################################################################
    # # fig, axs = plt.subplots(15, 2, figsize=(2 * 50, 15 * 20))
    # fig, axs = plt.subplots(5, 2, figsize=(2 * 50, 15 * 20))
    # ###############################################################################
    # # seq_features = feat_list[-1]
    eeg_atten_list = []  #
    eog_atten_list = []  #
    for i in range(num_seq):
        eeg_features = feat_list[i][0]  ##extracting learned intra-modal EEG features
        eog_features = feat_list[i][1]  ##extracting learned intra-modal EOG features
        cross_features = feat_list[i][-1]  ##extracting learned cross-modal features
        #
        eeg_atten = atten_interpret(cross_features.squeeze()[0].unsqueeze(0),
                                    eeg_features.squeeze()[1:])  # .squeeze().detach().cpu().numpy()
        eog_atten = atten_interpret(cross_features.squeeze()[0].unsqueeze(0),
                                    eog_features.squeeze()[1:])  # .squeeze().detach().cpu().numpy()
        #
        eeg_atten = F.upsample(eeg_atten.unsqueeze(0), scale_factor=3000 // 60,
                               mode='nearest').squeeze().detach().cpu().numpy()
        eog_atten = F.upsample(eog_atten.unsqueeze(0), scale_factor=3000 // 60,
                               mode='nearest').squeeze().detach().cpu().numpy()
        #
        eeg_atten_list.append(eeg_atten)  #
        eog_atten_list.append(eog_atten)  #
    #
    #     t1 = np.arange(0, 30, 1 / 256)
    #     plot_interpret(i, t1, eeg_data[t, 0, i, :].squeeze().cpu().numpy(), eeg_atten, fig, axs, [i, 0], signal_type="EEG")
    #     plot_interpret(i, t1, eog_data[t, 0, i, :].squeeze().cpu().numpy(), eog_atten, fig, axs, [i, 1], signal_type="EOG")
    #
    # save_dir = "static/picture"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # save_path = os.path.join(save_dir, "picture_4.jpg")
    # plt.savefig(save_path, dpi=200)
    #########################################################################################################################################


    """# Final plot similar to the paper"""
    # fig, axs = plt.subplots(num_seq, 4,figsize=(200, 20*num_seq),gridspec_kw={'width_ratios': [2,2,10,10]}) # for more clear figure
    fig, axs = plt.subplots(num_seq, 4, figsize=(120, 20 * num_seq), gridspec_kw={'width_ratios': [2, 2, 10, 10]})
    title_font_size = fig.dpi * 0.8
    label_font_size = fig.dpi * 0.6
    for i in range(num_seq):
        # Plotting inter-epoch attention ##############################
        rgba_colors = np.zeros((num_seq, 4))
        rgba_colors[:, 0] = 0  # value of red intensity divided by 256
        rgba_colors[i, 0] = 0.4  # value of red intensity divided by 256
        rgba_colors[:, 1] = 0  # value of green intensity divided by 256
        rgba_colors[:, 2] = 0.4  # value of blue intensity divided by 256
        rgba_colors[i, 2] = 0
        rgba_colors[:, -1] = seq_atten_list[i]
        axs[i][0].bar(np.arange(1, num_seq + 1), seq_atten_list[i] / seq_atten_list[i].max(),
                      # /seq_attn[i].max(),# color ='blue',
                      color=rgba_colors, align='center')
        # axs[i//5][i%5].set_title('')
        axs[i][0].tick_params(axis='x', labelsize=label_font_size)
        axs[i][0].tick_params(axis='y', labelsize=label_font_size)
        axs[i][0].set_xlabel('Epochs', fontsize=title_font_size)
        yticks = axs[i][0].yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)

        # Plotting cross-modal attention ##############################
        rgba_colors = np.zeros((2, 4))
        rgba_colors[:, 0] = 0.4  # value of red intensity divided by 256
        rgba_colors[:, 1] = 0  # value of green intensity divided by 256
        rgba_colors[:, 2] = 0  # value of blue intensity divided by 256
        rgba_colors[:, -1] = cross_atten_list[i]
        axs[i][1].bar(['EEG', 'EOG'], cross_atten_list[i],  # color ='red',
                      color=rgba_colors, align='center')
        axs[i][1].tick_params(axis='x', labelsize=label_font_size)
        axs[i][1].tick_params(axis='y', labelsize=label_font_size)
        axs[i][1].set_ylim(0, 1.02)
        axs[i][1].set_xlabel('Signal', fontsize=title_font_size)

        # # Plotting EEG attention ##############################
        eeg_atten_epoch = eeg_atten_list[i]
        t1 = np.arange(0, 3000, 1)
        plot_interpret(t1, eeg_data[t, 0, i, :].squeeze().cpu().numpy(), eeg_atten, fig, [i, 2],
                       signal_type=f"EEG Class:{pred_list[i]}")

        # plot_interpret(t1,eog_data[t,0,i,:].squeeze().cpu().numpy(),eog_atten,fig,[i,1],signal_type = "EOG")

        # # Plotting EOG attention ##############################
        eog_atten_epoch = eog_atten[i]
        plot_interpret(t1, eog_data[t, 0, i, :].squeeze().cpu().numpy(), eog_atten, fig, [i, 3],
                       signal_type=f"EOG Class:{pred_list[i]}")

    # time = [int(record_id.split('-')[1].split('_')[i]) for i in range(num_epoch_seq)]
    # plt.subplots_adjust(wspace=0.2)
    # fig.suptitle('Interpretation for patient '+str([38])+' for 30s epochs from '+str(start_time_point)+'s',fontsize = title_font_size*2)


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_path = os.path.join(out_dir, "interpretation.jpg")
    plt.savefig(save_path, dpi=150)

    return result_dict, pred_stages
#if __name__ == '__main__':
 #   main()
