import pyedflib
import numpy as np

import torch
from model.model_run_util import SleepEDF_Seq_MultiChan_Dataset_Inference
from data_preparations.new_single_epoch_Sleep_EDF_153 import Sleep_EDF_SC_signal_extract_WITHOUT_HY

from torchvision import transforms, datasets
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

def fetch_seq(real_file,model_config):
    eeg_raw_data, eeg_sub_len, eeg_mean, eeg_std = (
        Sleep_EDF_SC_signal_extract_WITHOUT_HY(real_file, channel="eeg1", filter=True, freq=[0.2, 40])
    )
    eog_raw_data, _, eog_mean, eog_std = (
        Sleep_EDF_SC_signal_extract_WITHOUT_HY(real_file, channel="eog", filter=True, freq=[0.2, 40])
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_seq = model_config["use"]["num_seq"]
    # num_seq = 5

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

    eeg_data_temp = eeg_data[0].squeeze()  # (0)
    eog_data_temp = eog_data[0].squeeze()  # (0)

    print(eeg_data_temp[0].squeeze())
    print(eog_data_temp[0].squeeze())

    return eeg_data_temp[0].squeeze(), eog_data_temp[0].squeeze()