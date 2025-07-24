'''
除了原有的预处理signal_extract外
还加入了针对目前遇到的数据集的不同预处理接口
Sleep-EDF, DREAMS, Figshare, hospital
'''


#path必须以\\结尾，否则传入的路径与文件名拼接错误
data_path = r"E:\\hkk\\项目_可解释睡眠分期\\项目原始数据集\\physionet-sleep-data\\"
save_path = r"E:\\hkk\\项目_可解释睡眠分期\\项目预处理后数据\\Singel预处理_Sleep-EDF-153_eeg1_eog1\\"


# 导入库
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import mne
from mne.datasets.sleep_physionet.age import fetch_data
# from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import h5py
import argparse
import pyedflib
import pandas as pd
from datetime import datetime, timedelta



# 用于接收控制台提供的参数
def parse_option():
    parser = argparse.ArgumentParser('Argument for data generation')
    parser.add_argument('--save_path', type=str, default='./extract_dataset_single_epoch',
                        help='Path to store project results')

    opt = parser.parse_args()
    return opt



# 针对Sleep_EDF数据集，需要输入<受试者编号数组subjects[]>和<天数数组days[]>
# 返回PSG信号数组，标签二维数组，PSG信号长度数组，均值数组，标准差数组
def signal_extract(subjects, days, channel='eeg1', filter=True, freq=[0.2, 40]):
    # 需要忽略的数据
    ignore_data = [[13, 2], [36, 1], [39, 1], [39, 2], [52, 1], [68, 1], [68, 2], [69, 1], [69, 2], [78, 1], [78, 2],
                   [79, 1], [79, 2]]
    # 所有通道
    all_channels = ('EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Resp oro-nasal', 'Temp rectal', 'Event marker')

    # 标记是否是第一份数据
    first_sub_flag = 0

    # 遍历所有数据并保存
    for sub in subjects:
        for day_ in days:
            if [sub, day_] in ignore_data:
                continue
            [data] = fetch_data(subjects=[sub], recording=[day_])
            signal2idx = {"eeg1": 0, "eeg2": 1, "eog": 2, "emg": 3}

            all_channels_list = list(all_channels)
            all_channels_list.remove(all_channels[signal2idx[channel]])
            exclude_channels = tuple(all_channels_list)

            sleep_signals = mne.io.read_raw_edf(data[0], verbose=True, exclude=exclude_channels, preload=True)

            annot = mne.read_annotations(data[1])

            # 睡眠阶段和数字的对应关系
            ann2label = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
                         "Sleep stage 4": 4, "Sleep stage R": 5}
            #     # "Sleep stage ?": 5,
            #     # "Movement time": 5

            # 切割
            annot.crop(annot[1]['onset'] - 30 * 60,
                       annot[-2]['onset'] + 30 * 60)

            sleep_signals.set_annotations(annot, emit_warning=False)

            events, _ = mne.events_from_annotations(
                sleep_signals, event_id=ann2label, chunk_duration=30.)

            # Filtering
            tmax = 30. - 1. / sleep_signals.info['sfreq']

            if filter == True:
                sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

            # Breaking into Epochs
            epochs_data = mne.Epochs(raw=sleep_signals, events=events,
                                     event_id=ann2label, tmin=0., tmax=tmax, baseline=None, preload=True,
                                     on_missing='warn')

            sig_epochs = []
            label_epochs = []

            mean_epochs = []
            std_epochs = []

            signal_mean = np.mean(np.array([epochs_data]))
            signal_std = np.std(np.array([epochs_data]))

            for ep in range(len(epochs_data)):
                for sig in epochs_data[ep]:
                    sig_epochs.append(sig)

                sleep_stage = epochs_data[ep].event_id

                if sleep_stage == {"Sleep stage W": 0}:
                    label_epochs.append(0)
                if sleep_stage == {"Sleep stage 1": 1}:
                    label_epochs.append(1)
                if sleep_stage == {"Sleep stage 2": 2}:
                    label_epochs.append(2)
                if sleep_stage == {"Sleep stage 3": 3}:
                    label_epochs.append(3)
                if sleep_stage == {"Sleep stage 4": 4}:
                    label_epochs.append(3)
                if sleep_stage == {"Sleep stage R": 5}:
                    label_epochs.append(4)

                mean_epochs.append(signal_mean)
                std_epochs.append(signal_std)

            sig_epochs = np.array(sig_epochs)
            mean_epochs = np.array(mean_epochs)
            std_epochs = np.array(std_epochs)
            label_epochs = np.array(label_epochs)

            if first_sub_flag == 0:
                main_ext_raw_data = sig_epochs
                main_labels = label_epochs
                main_sub_len = np.array([len(epochs_data)])
                main_mean = mean_epochs
                main_std = std_epochs
                first_sub_flag = 1
            else:
                main_ext_raw_data = np.concatenate((main_ext_raw_data, sig_epochs), axis=0)
                main_labels = np.concatenate((main_labels, label_epochs), axis=0)
                main_sub_len = np.concatenate((main_sub_len, np.array([len(epochs_data)])), axis=0)
                main_mean = np.concatenate((main_mean, mean_epochs), axis=0)
                main_std = np.concatenate((main_std, std_epochs), axis=0)

    return main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std



# 针对Sleep_EDF数据集，需要输入<psg信号文件地址psg_file>和<注释文件地址annotation_file>
# 返回psg信号，标签数组，psg信号长度，均值，标准差
# 这个数据预处理切割时间从熄灯到倒数第二个注释
def Sleep_EDF_SC_signal_extract(files, channel='eeg1', filter=True, freq=[0.2, 40]):
    # 所有通道
    all_channels = (
        'EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Resp oro-nasal', 'Temp rectal', 'Event marker')

    first_sub_flag = 0
    for file in files:
        data = [data_path+file[0], data_path+file[1]]
        signal2idx = {"eeg1": 0, "eeg2": 1, "eog": 2, "emg": 3}

        all_channels_list = list(all_channels)
        all_channels_list.remove(all_channels[signal2idx[channel]])
        exclude_channels = tuple(all_channels_list)

        sleep_signals = mne.io.read_raw_edf(data[0], verbose=True, exclude=exclude_channels, preload=True)

        annot = mne.read_annotations(data[1])

        ann2label = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
                     "Sleep stage 4": 4, "Sleep stage R": 5}

        #########################
        # print("------------------------------", annot[1], "---------------------------------")
        # data_sleep_light_of_time = pd.read_excel("D:\hkk\项目_可解释性睡眠分期\Transformer\原始数据集\sleep-edf-database-expanded-1.0.0\SC-subjects.xlsx")
        # T_Light_off = data_sleep_light_of_time[['LightsOff']]
        #########################

        # sc4_index = annotation_file.find("SC4")
        # 从"SC4"的位置开始，提取后面三个字符
        # if sc4_index != -1:
        #     extracted_chars = annotation_file[sc4_index + 3:sc4_index + 6]
        #     print(extracted_chars)

        # 获取到受试者及天数信息，通过light_off_time函数获取熄灯时间
        # sub = int(extracted_chars[:2])
        # day = int(extracted_chars[2])
        # start_to_light_off = light_off_time(sub, day, annotation_file)
        # print(sub)
        # print(day)
        # print("annot[1][oneset] = ", annot[1]['onset'])

        # 按熄灯时间切割
        # 目前受试者的起床时间还没有确定，所以采用倒数第二个event的时间

        annot.crop(annot[1]['onset'] - 30 * 60,
                annot[-2]['onset'] + 30 * 60)

        sleep_signals.set_annotations(annot, emit_warning=False)

        events, _ = mne.events_from_annotations(
            sleep_signals, event_id=ann2label, chunk_duration=30.)

        # Filtering
        tmax = 30. - 1. / sleep_signals.info['sfreq']

        if filter == True:
            sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

        # Breaking into Epochs
        epochs_data = mne.Epochs(raw=sleep_signals, events=events,
                                 event_id=ann2label, tmin=0., tmax=tmax, baseline=None, preload=True,
                                 on_missing='warn')

        sig_epochs = []
        label_epochs = []

        mean_epochs = []
        std_epochs = []

        signal_mean = np.mean(np.array([epochs_data]))
        signal_std = np.std(np.array([epochs_data]))

        for ep in range(len(epochs_data)):
            for sig in epochs_data[ep]:
                sig_epochs.append(sig)

            sleep_stage = epochs_data[ep].event_id

            if sleep_stage == {"Sleep stage W": 0}:
                label_epochs.append(0)
            if sleep_stage == {"Sleep stage 1": 1}:
                label_epochs.append(1)
            if sleep_stage == {"Sleep stage 2": 2}:
                label_epochs.append(2)
            if sleep_stage == {"Sleep stage 3": 3}:
                label_epochs.append(3)
            if sleep_stage == {"Sleep stage 4": 4}:
                label_epochs.append(3)
            if sleep_stage == {"Sleep stage R": 5}:
                label_epochs.append(4)

            mean_epochs.append(signal_mean)
            std_epochs.append(signal_std)

        sig_epochs = np.array(sig_epochs)
        mean_epochs = np.array(mean_epochs)
        std_epochs = np.array(std_epochs)
        label_epochs = np.array(label_epochs)

        if first_sub_flag == 0:
            main_ext_raw_data = sig_epochs
            main_labels = label_epochs
            main_sub_len = np.array([len(epochs_data)])
            main_mean = mean_epochs
            main_std = std_epochs
            first_sub_flag = 1
        else:
            main_ext_raw_data = np.concatenate((main_ext_raw_data, sig_epochs), axis=0)
            main_labels = np.concatenate((main_labels, label_epochs), axis=0)
            main_sub_len = np.concatenate((main_sub_len, np.array([len(epochs_data)])), axis=0)
            main_mean = np.concatenate((main_mean, mean_epochs), axis=0)
            main_std = np.concatenate((main_std, std_epochs), axis=0)

    return main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std


# 输入<受试者编号sub>,<天数day>和<注释文件地址edf_file_path>
# 返回熄灯时间和开始记录时间的差值
def light_off_time(sub, day, edf_file_path):
    # 示例数据
    # sub = 0
    # day = 1
    # edf_file_path = "D:\\hkk\\项目_可解释性睡眠分期\\Transformer\\原始数据集\\sleep-edf-database-expanded-1.0.0\\sleep-cassette\\SC4001EC-Hypnogram.edf"

    # 记录sleep-cassette受试者熄灯时间的excel文件--SC-subjects.xlsx(更改为自己的地址即可)
    excel_path = "D:\\hkk\\项目_可解释性睡眠分期\\Transformer\\原始数据集\\sleep-edf-database-expanded-1.0.0\\SC-subjects.xlsx"

    # 打开EDF文件
    f = pyedflib.EdfReader(edf_file_path)

    # 开始记录PSG的时间
    start_time = f.getStartdatetime()
    # print("start_record_time:", start_time)

    # 受试者熄灯时间
    data_sleep_light_of_time = pd.read_excel(excel_path)
    filtered_data = data_sleep_light_of_time[(data_sleep_light_of_time['subject'] == sub) & (data_sleep_light_of_time['night'] == day)]
    # 确保 lights_off_time 是字符串
    lights_off_time_str = filtered_data['LightsOff'].iloc[0]
    # print("lights_of_time:",lights_off_time_str)

    # 将字符串转换为 datetime.time 对象
    # lights_off_time_obj = datetime.strptime(lights_off_time_str, '%H:%M:%S').time()

    # 创建完整的日期时间对象
    lights_off_datetime = datetime.combine(start_time.date(), lights_off_time_str)

    # 如果 lights_off_time 在第二天，需要调整日期
    if lights_off_time_str < start_time.time():
        lights_off_datetime += timedelta(days=1)

    # 计算时间差
    time_difference = lights_off_datetime - start_time

    # print("time_diffence(h):",time_difference)
    # print("time_diffence(s):",time_difference.total_seconds())

    f.close()

    return time_difference.total_seconds()

def Sleep_EDF_SC_signal_extract_WITHOUT_HY(psg_file, channel='eeg1', filter=True, freq=[0.2, 40]):
    all_channels = (
    'EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Resp oro-nasal', 'Temp rectal', 'Event marker')

    data = [psg_file]
    signal2idx = {"eeg1": 0, "eeg2": 1, "eog": 2, "emg": 3}

    all_channels_list = list(all_channels)
    all_channels_list.remove(all_channels[signal2idx[channel]])
    exclude_channels = tuple(all_channels_list)

    sleep_signals = mne.io.read_raw_edf(data[0], verbose=True, exclude=exclude_channels, preload=True)

    # Filtering
    tmax = 30. - 1. / sleep_signals.info['sfreq']

    if filter == True:
        sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

    duration = 30
    epochs = mne.make_fixed_length_epochs(sleep_signals, duration=duration, preload=True)

    # Calculate mean and std of the signal epochs
    signal_mean = np.mean(epochs)
    signal_std = np.std(epochs)

    main_ext_raw_data = epochs.get_data()
    main_sub_len = np.array([len(epochs)])
    main_mean = np.tile(signal_mean, (len(epochs), 1)).squeeze()
    main_std = np.tile(signal_std, (len(epochs), 1)).squeeze()

    return main_ext_raw_data, main_sub_len, main_mean, main_std

# 针对Dreams数据集，需要输入<PSG信号地址psg_path>和<注释文件地址hypnogram_path>
# 返回psg信号，标签数组，psg信号长度，均值，标准差
# 需要注意原本的注释文件数字对应的睡眠阶段和Sleep-edf是不一样的，但是我们事先做了注释转换，所以此处代码的对应关系不需要改变
def DREAMS_signal_extract(psg_path, hypnogram_path, channel='FP1-A2', filter=True, freq=[0.2, 40]):
    id_idx1 = psg_path.find("subject")
    id_idx2 = psg_path.find(".edf")
    id = psg_path[id_idx1+7:id_idx2]
    id = int(id)
    if id != 2 and id != 6 and id != 9:
        all_channels = ('ECG', 'FP1-A2', 'CZ-A1', 'EMG1', 'EOG1', 'VTH', 'VAB',
                        'NAF2P-A1', 'NAF1', 'PHONO', 'PR', 'SAO2', 'PCPAP', 'POS',
                        'EOG2', 'O1-A2', 'FP2-A1', 'O2-A1', 'CZ2-A1', 'EMG2',
                        'PULSE', 'VTOT', 'EMG3')
    else:
        # DREAMS数据集2,6,9受试者专属
        all_channels = ('ECG', 'FP1-A2', 'CZ-A1', 'EMG1', 'EOG1-A2', 'VTH', 'VAB',
                        'NAF2P-A1', 'NAF1', 'PHONO', 'PR', 'SAO2', 'PCPAP', 'POS',
                        'EOG2-A2', 'O1-A2', 'FP2-A1', 'O2-A1', 'CZ2-A1', 'EMG2',
                        'PULSE', 'VTOT', 'EMG3')


    data = [psg_path, hypnogram_path]
    print("preparing: " + data[0] + " " + data[1])
    if id != 2 and id != 6 and id != 9:
        signal2idx = {"ECG": 0, "FP1-A2": 1, "CZ-A1": 2, "EMG1": 3, "EOG1" : 4}
    else:
        signal2idx = {"ECG": 0, "FP1-A2": 1, "CZ-A1": 2, "EMG1": 3, "EOG1-A2": 4}

    all_channels_list = list(all_channels)
    all_channels_list.remove(all_channels[signal2idx[channel]])
    exclude_channels = tuple(all_channels_list)

    # print(exclude_channels)
    # print(len(exclude_channels))

    sleep_signals = mne.io.read_raw_edf(data[0], verbose=True, exclude=list(exclude_channels), preload=True)

    annot = mne.read_annotations(data[1])

    ann2label = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
                 "Sleep stage 4": 4, "Sleep stage R": 5}
    #     # "Sleep stage ?": 5,
    #     # "Movement time": 5

    # 没有对DREAMS数据集进行数据切割
    '''
    annot.crop(annot[1]['onset'] - 30 * 60,
               annot[-2]['onset'] + 30 * 60)
    '''

    sleep_signals.set_annotations(annot, emit_warning=False)

    events, _ = mne.events_from_annotations(
        sleep_signals, event_id=ann2label, chunk_duration=30.)

    # Filtering
    tmax = 30. - 1. / sleep_signals.info['sfreq']

    if filter == True:
        sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

    # Breaking into Epochs
    epochs_data = mne.Epochs(raw=sleep_signals, events=events,
                             event_id=ann2label, tmin=0., tmax=tmax, baseline=None, preload=True,
                             on_missing='warn')
    # epochs = mne.make_fixed_length_epochs(sleep_signals, duration=duration, preload=True)

    sig_epochs = []
    label_epochs = []

    mean_epochs = []
    std_epochs = []

    signal_mean = np.mean(np.array([epochs_data]))
    signal_std = np.std(np.array([epochs_data]))

    for ep in range(len(epochs_data)):
        for sig in epochs_data[ep]:
            sig_epochs.append(sig)

        sleep_stage = epochs_data[ep].event_id

        if sleep_stage == {"Sleep stage W": 9}:
            label_epochs.append(0)
        if sleep_stage == {"Sleep stage 1": 1}:
            label_epochs.append(1)
        if sleep_stage == {"Sleep stage 2": 2}:
            label_epochs.append(2)
        if sleep_stage == {"Sleep stage 3": 3}:
            label_epochs.append(3)
        if sleep_stage == {"Sleep stage 4": 4}:
            label_epochs.append(3)
        if sleep_stage == {"Sleep stage R": 5}:
            label_epochs.append(4)

        mean_epochs.append(signal_mean)
        std_epochs.append(signal_std)

    sig_epochs = np.array(sig_epochs)
    mean_epochs = np.array(mean_epochs)
    std_epochs = np.array(std_epochs)
    label_epochs = np.array(label_epochs)

    main_ext_raw_data = sig_epochs
    main_labels = label_epochs
    main_sub_len = np.array([len(epochs_data)])
    main_mean = mean_epochs
    main_std = std_epochs

    return main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std



# 针对figshare数据集，需要输入<PSG信号文件地址subjects>
# 返回PSG信号，长度，均值，标准差
def figshare_signal_extract(subjects, channel='eeg1', filter=True, freq=[0.2, 40]):
    all_channels = (
        'EEG Fp1-LE', 'EEG F3-LE', 'EEG C3-LE', 'EEG P3-LE', 'EEG O1-LE', 'EEG F7-LE', 'EEG T3-LE',
        'EEG T5-LE', 'EEG Fz-LE', 'EEG Fp2-LE', 'EEG F4-LE', 'EEG C4-LE', 'EEG P4-LE', 'EEG O2-LE',
        'EEG F8-LE', 'EEG T4-LE', 'EEG T6-LE', 'EEG Cz-LE', 'EEG Pz-LE', 'EEG A2-A1', 'EEG 23A-23R',
        'EEG 24A-24R')

    data = [subjects]
    # eeg1代表'EEG Fz-LE'
    # eog代表'EEG Pz-LE'
    signal2idx = {"eeg1": 8, "eeg2": 1, "eog": 18, "emg": 3}

    all_channels_list = list(all_channels)
    all_channels_list.remove(all_channels[signal2idx[channel]])
    exclude_channels = tuple(all_channels_list)

    sleep_signals = mne.io.read_raw_edf(data[0], verbose=True, exclude=exclude_channels, preload=True)

    # Filtering
    # tmax = 30. - 1. / sleep_signals.info['sfreq']

    if filter == True:
        sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

    duration = 30
    epochs = mne.make_fixed_length_epochs(sleep_signals, duration=duration, preload=True)

    # Calculate mean and std of the signal epochs
    signal_mean = np.mean(epochs)
    signal_std = np.std(epochs)

    main_ext_raw_data = epochs.get_data()
    main_sub_len = np.array([len(epochs)])
    main_mean = np.tile(signal_mean, (len(epochs), 1)).squeeze()
    main_std = np.tile(signal_std, (len(epochs), 1)).squeeze()

    return main_ext_raw_data, main_sub_len, main_mean, main_std


# 针对未经过预处理的hospital数据集，需要输入
def signal_extract_hospital(edf_anno_pairs, channel='eeg1', filter=True, freq=[0.2, 40], stride=3):

# 1.初始化无效数据、通道
    ignore_data = []

#【改通道】
    all_channels = (
        'F3', 'F4', 'C3', 'C4', 'O1', 'O2',
        'M1', 'M2',
        'E1', 'E2',
        'ECG1', 'ECG2',
        'Chin1', 'Chin2', 'Chin3', 'LEG/L', 'LEG/R',
        'Airflow', 'Abdo', 'Thor', 'Snore', 'Sum', 'PosSensor', 'Ox Status', 'Pulse', 'SpO2', 'Nasal Pressure', 'CPAP Flow',
        'CPAP Press', 'Pleth', 'Sum', 'Derived HR', 'Light', 'Manual Pos', 'Respiratory Rate'
    )

    first_sub_flag = 0

    for pair in edf_anno_pairs:
            data = [ hospital_path + pair[0], hospital_path + pair[1]]
            print("preparing: " + data[0] + " " + data[1])

        # 【改数据获取】
            signal2idx = {"eeg1": 0, "eeg2": 1, "eeg3": 2, "eeg4": 3, "eeg5": 4, "eeg6": 5,
                          "eog1": 7, "eog2": 8}
            all_channels_list = list(all_channels)
            all_channels_list.remove(all_channels[signal2idx[channel]])
            exclude_channels = tuple(all_channels_list)

            sleep_signals = mne.io.read_raw_edf(data[0], verbose=True, exclude=exclude_channels, preload=True)
            annot = mne.read_annotations(data[1])
            # print("Annotation descriptions:", annot.description)

        # 3.注释裁剪和事件生成
        # 【改映射】
        #     ann2label = {
        #         "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
        #          "Sleep stage R": 4, "Sleep stage ?": 5, "Movement time": 6}
            ann2label = {
                "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3, "Sleep stage R": 4}

            ann2label_without_unknown_stages = {
                "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3, "Sleep stage R": 4}

            # annot.crop(annot[1]['onset'] - 30 * 60, annot[-2]['onset'] + 30 * 60)

            sleep_signals.set_annotations(annot, emit_warning=False)

            events, _ = mne.events_from_annotations(
                sleep_signals, event_id=ann2label, chunk_duration=30.)


        # 4.信号过滤
            if filter == True:
                sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

        # 5.划分 Epoch
            tmax = 30. - 1. / sleep_signals.info['sfreq']
            epochs_data = mne.Epochs(raw=sleep_signals, events=events,
                                     event_id=ann2label, tmin=0., tmax=tmax, baseline=None, preload=True,
                                     on_missing='warn')

            epochs_data_without_unknown_stages = mne.Epochs(raw=sleep_signals, events=events,
                                                            event_id=ann2label_without_unknown_stages, tmin=0.,
                                                            tmax=tmax, baseline=None, preload=True, on_missing='warn')

            print(
                '===================================================================================================================================')
            print(
                f"                    Shape of Extracted Raw Signal for File {pair}                           ")
            print(
                f"                    Shape of Extracted Label for File {pair}                             ")
            # print('===================================================================================================================================')

            sig_epochs = []
            label_epochs = []

            mean_epochs = []
            std_epochs = []

            signal_mean = np.mean(np.array([epochs_data]))
            signal_std = np.std(np.array([epochs_data]))

            for ep in range(len(epochs_data)):
                for sig in epochs_data[ep]:
                    sig_epochs.append(sig)

                sleep_stage = epochs_data[ep].event_id

                if sleep_stage == {"Sleep stage W": 0}:
                    label_epochs.append(0)
                if sleep_stage == {"Sleep stage 1": 1}:
                    label_epochs.append(1)
                if sleep_stage == {"Sleep stage 2": 2}:
                    label_epochs.append(2)
                if sleep_stage == {"Sleep stage 3": 3}:
                    label_epochs.append(3)
                if sleep_stage == {"Sleep stage R": 4}:
                    label_epochs.append(4)

                mean_epochs.append(signal_mean)
                std_epochs.append(signal_std)

            sig_epochs = np.array(sig_epochs)
            mean_epochs = np.array(mean_epochs)
            std_epochs = np.array(std_epochs)
            label_epochs = np.array(label_epochs)

            if first_sub_flag == 0:
                main_ext_raw_data = sig_epochs
                main_labels = label_epochs
                main_sub_len = np.array([len(epochs_data)])
                main_mean = mean_epochs
                main_std = std_epochs
                first_sub_flag = 1
            else:
                main_ext_raw_data = np.concatenate((main_ext_raw_data, sig_epochs), axis=0)
                main_labels = np.concatenate((main_labels, label_epochs), axis=0)
                main_sub_len = np.concatenate((main_sub_len, np.array([len(epochs_data)])), axis=0)
                main_mean = np.concatenate((main_mean, mean_epochs), axis=0)
                main_std = np.concatenate((main_std, std_epochs), axis=0)

    return main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std


# 针对未经过预处理的hospital数据集，需要输入
def signal_extract_hospital_processed(edf_anno_pairs, channel='eeg1', filter=True, freq=[0.2, 40]):

# 1.初始化无效数据、通道
    ignore_data = []

#【改通道】
    all_channels = (
        'F3', 'F4', 'C3', 'C4', 'O1', 'O2',
        'M1', 'M2',
        'E1', 'E2',
    )

    first_sub_flag = 0

    for pair in edf_anno_pairs:
            data = [ hospital_path + pair[0], hospital_path + pair[1]]
            print("preparing: " + data[0] + " " + data[1])

        # 【改数据获取】
            signal2idx = {"eeg1": 0, "eeg2": 1, "eeg3": 2, "eeg4": 3, "eeg5": 4, "eeg6": 5,
                          "eog1": 8, "eog2": 9}
            all_channels_list = list(all_channels)
            all_channels_list.remove(all_channels[signal2idx[channel]])
            exclude_channels = tuple(all_channels_list)

            sleep_signals = mne.io.read_raw_edf(data[0], verbose=True, exclude=exclude_channels, preload=True)
            annot = mne.read_annotations(data[1])
            # print("Annotation descriptions:", annot.description)

        # 3.注释裁剪和事件生成
        # 【改映射】
        #     ann2label = {
        #         "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
        #          "Sleep stage R": 4, "Sleep stage ?": 5, "Movement time": 6}
            ann2label = {
                "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3, "Sleep stage R": 4}

            # ann2label_without_unknown_stages = {
            #     "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3, "Sleep stage R": 4}

            # annot.crop(annot[1]['onset'] - 30 * 60, annot[-2]['onset'] + 30 * 60)

            sleep_signals.set_annotations(annot, emit_warning=False)


            events, _ = mne.events_from_annotations(
                sleep_signals, event_id=ann2label, chunk_duration=30.)


        # 4.信号过滤
            if filter == True:
                sleep_signals = sleep_signals.copy().filter(l_freq=freq[0], h_freq=freq[1])

        # 5.划分 Epoch
            tmax = 30. - 1. / sleep_signals.info['sfreq']
            epochs_data = mne.Epochs(raw=sleep_signals, events=events,
                                     event_id=ann2label, tmin=0., tmax=tmax, baseline=None, preload=True,
                                     on_missing='warn')

            # epochs_data_without_unknown_stages = mne.Epochs(raw=sleep_signals, events=events,
            #                                                 event_id=ann2label_without_unknown_stages, tmin=0.,
            #                                                 tmax=tmax, baseline=None, preload=True, on_missing='warn')

            print(
                '===================================================================================================================================')
            print(
                f"                    Shape of Extracted Raw Signal for File {pair}                           ")
            print(
                f"                    Shape of Extracted Label for File {pair}                             ")
            print('===================================================================================================================================')

            sig_epochs = []
            label_epochs = []

            mean_epochs = []
            std_epochs = []

            signal_mean = np.mean(np.array([epochs_data]))
            signal_std = np.std(np.array([epochs_data]))

            for ep in range(len(epochs_data)):
                for sig in epochs_data[ep]:
                    sig_epochs.append(sig)

                sleep_stage = epochs_data[ep].event_id

                if sleep_stage == {"Sleep stage W": 0}:
                    label_epochs.append(0)
                if sleep_stage == {"Sleep stage 1": 1}:
                    label_epochs.append(1)
                if sleep_stage == {"Sleep stage 2": 2}:
                    label_epochs.append(2)
                if sleep_stage == {"Sleep stage 3": 3}:
                    label_epochs.append(3)
                if sleep_stage == {"Sleep stage R": 4}:
                    label_epochs.append(4)

                mean_epochs.append(signal_mean)
                std_epochs.append(signal_std)

            sig_epochs = np.array(sig_epochs)
            mean_epochs = np.array(mean_epochs)
            std_epochs = np.array(std_epochs)
            label_epochs = np.array(label_epochs)

            if first_sub_flag == 0:
                main_ext_raw_data = sig_epochs
                main_labels = label_epochs
                main_sub_len = np.array([len(epochs_data)])
                main_mean = mean_epochs
                main_std = std_epochs
                first_sub_flag = 1
            else:
                main_ext_raw_data = np.concatenate((main_ext_raw_data, sig_epochs), axis=0)
                main_labels = np.concatenate((main_labels, label_epochs), axis=0)
                main_sub_len = np.concatenate((main_sub_len, np.array([len(epochs_data)])), axis=0)
                main_mean = np.concatenate((main_mean, mean_epochs), axis=0)
                main_std = np.concatenate((main_std, std_epochs), axis=0)

    return main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std


# 主函数，封装数据为h5py文件保存
def main():
    from sklearn.model_selection import KFold
    import fnmatch
    # args = parse_option()
    # if args.save_path !="": save_path  = args.save_path
    # if args.data_path !="": hospital_path  = args.data_path
    # 需要处理的通道
    channels = ["eeg1", "eog"]

    edf_anno_list = []
    # 查找所有的edf文件和其对应的注释文件【列表edf_anno_list】
    for filename in os.listdir(data_path):
        if fnmatch.fnmatch(filename, '*-Hypnogram.edf'):  # 查找以 "?" 结尾的文件
            tmp = filename.split('*-Hypnogram.edf')[0]  # 提取文件名除去"?"的前缀部分
            tmp = tmp[3:6]
            # print(tmp)
            for i in os.listdir(data_path):
                if fnmatch.fnmatch(i, 'SC4' + tmp + 'E0-PSG.edf'):  # 查找以相同前缀并以 .edf 结尾的文件
                    edf_anno_list.append((i, filename))  # 将edf文件、注释文件的元组添加到列表中(PSG文件, 注释文件)
                    print(i + " matched " + filename)
    print(f"Found {len(edf_anno_list)} EDF and annotation file pairs.")
    print("===================================================================================")

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=2)

    # 存储每个fold的测试集文件对（直接存储edf_anno_list的子集）
    fold_subsets = []
    for train_idx, test_idx in kf.split(edf_anno_list):
        fold_subsets.append([edf_anno_list[i] for i in test_idx])  # 存储实际文件对

    # 解包到sub_1, sub_2等变量
    sub_1, sub_2, sub_3, sub_4, sub_5 = fold_subsets

    # 打印验证
    print(f"Subjects Group 1 : {sub_1}")
    print(f"Subjects Group 2 : {sub_2}")
    print(f"Subjects Group 3 : {sub_3}")
    print(f"Subjects Group 4 : {sub_4}")
    print(f"Subjects Group 5 : {sub_5}")


    label_saved = False
    for channel in channels:
        main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std = (
                Sleep_EDF_SC_signal_extract(sub_1, channel=channel, filter=True, freq=[0.2, 40])
            )

        if "eeg1" in channel:
            # 保存信号数据
            with h5py.File(f'{save_path}/eeg1_x1.h5', 'w') as f:
                f.create_dataset(channel, data=main_ext_raw_data)

            # 保存均值
            with h5py.File(f'{save_path}/eeg1_mean1.h5', 'w') as f:
                f.create_dataset('mean', data=main_mean)

            # 保存标准差
            with h5py.File(f'{save_path}/eeg1_std1.h5', 'w') as f:
                f.create_dataset('std', data=main_std)

        if "eog" in channel:
            # 保存信号数据
            with h5py.File(f'{save_path}/eog1_x1.h5', 'w') as f:
                f.create_dataset(channel, data=main_ext_raw_data)

            # 保存均值
            with h5py.File(f'{save_path}/eog1_mean1.h5', 'w') as f:
                f.create_dataset('mean', data=main_mean)

            # 保存标准差
            with h5py.File(f'{save_path}/eog1_std1.h5', 'w') as f:
                f.create_dataset('std', data=main_std)

        # 保存标签数据（只保存一次）
        if not label_saved:
            with h5py.File(f'{save_path}/labels1.h5', 'w') as f:
                f.create_dataset('labels', data=main_labels)
            label_saved = True

    label_saved = False
    for channel in channels:
        main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std = (
            Sleep_EDF_SC_signal_extract(sub_2, channel=channel, filter=True, freq=[0.2, 40])
        )

        if "eeg1" in channel:
            # 保存信号数据
            with h5py.File(f'{save_path}/eeg1_x2.h5', 'w') as f:
                f.create_dataset(channel, data=main_ext_raw_data)

            # 保存均值
            with h5py.File(f'{save_path}/eeg1_mean2.h5', 'w') as f:
                f.create_dataset('mean', data=main_mean)

            # 保存标准差
            with h5py.File(f'{save_path}/eeg1_std2.h5', 'w') as f:
                f.create_dataset('std', data=main_std)

        if "eog" in channel:
            # 保存信号数据
            with h5py.File(f'{save_path}/eog1_x2.h5', 'w') as f:
                f.create_dataset(channel, data=main_ext_raw_data)

            # 保存均值
            with h5py.File(f'{save_path}/eog1_mean2.h5', 'w') as f:
                f.create_dataset('mean', data=main_mean)

            # 保存标准差
            with h5py.File(f'{save_path}/eog1_std2.h5', 'w') as f:
                f.create_dataset('std', data=main_std)

        # 保存标签数据（只保存一次）
        if not label_saved:
            with h5py.File(f'{save_path}/labels2.h5', 'w') as f:
                f.create_dataset('labels', data=main_labels)
            label_saved = True

    label_saved = False
    for channel in channels:
        main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std = (
            Sleep_EDF_SC_signal_extract(sub_3, channel=channel, filter=True, freq=[0.2, 40])
        )

        if "eeg1" in channel:
            # 保存信号数据
            with h5py.File(f'{save_path}/eeg1_x3.h5', 'w') as f:
                f.create_dataset(channel, data=main_ext_raw_data)

            # 保存均值
            with h5py.File(f'{save_path}/eeg1_mean3.h5', 'w') as f:
                f.create_dataset('mean', data=main_mean)

            # 保存标准差
            with h5py.File(f'{save_path}/eeg1_std3.h5', 'w') as f:
                f.create_dataset('std', data=main_std)

        if "eog" in channel:
            # 保存信号数据
            with h5py.File(f'{save_path}/eog1_x3.h5', 'w') as f:
                f.create_dataset(channel, data=main_ext_raw_data)

            # 保存均值
            with h5py.File(f'{save_path}/eog1_mean3.h5', 'w') as f:
                f.create_dataset('mean', data=main_mean)

            # 保存标准差
            with h5py.File(f'{save_path}/eog1_std3.h5', 'w') as f:
                f.create_dataset('std', data=main_std)

        # 保存标签数据（只保存一次）
        if not label_saved:
            with h5py.File(f'{save_path}/labels3.h5', 'w') as f:
                f.create_dataset('labels', data=main_labels)
            label_saved = True

    label_saved = False
    for channel in channels:
        main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std = (
            Sleep_EDF_SC_signal_extract(sub_4, channel=channel, filter=True, freq=[0.2, 40])
        )

        if "eeg1" in channel:
            # 保存信号数据
            with h5py.File(f'{save_path}/eeg1_x4.h5', 'w') as f:
                f.create_dataset(channel, data=main_ext_raw_data)

            # 保存均值
            with h5py.File(f'{save_path}/eeg1_mean4.h5', 'w') as f:
                f.create_dataset('mean', data=main_mean)

            # 保存标准差
            with h5py.File(f'{save_path}/eeg1_std4.h5', 'w') as f:
                f.create_dataset('std', data=main_std)

        if "eog" in channel:
            # 保存信号数据
            with h5py.File(f'{save_path}/eog1_x4.h5', 'w') as f:
                f.create_dataset(channel, data=main_ext_raw_data)

            # 保存均值
            with h5py.File(f'{save_path}/eog1_mean4.h5', 'w') as f:
                f.create_dataset('mean', data=main_mean)

            # 保存标准差
            with h5py.File(f'{save_path}/eog1_std4.h5', 'w') as f:
                f.create_dataset('std', data=main_std)

        # 保存标签数据（只保存一次）
        if not label_saved:
            with h5py.File(f'{save_path}/labels4.h5', 'w') as f:
                f.create_dataset('labels', data=main_labels)
            label_saved = True

    label_saved = False
    for channel in channels:
        main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std = (
            Sleep_EDF_SC_signal_extract(sub_5, channel=channel, filter=True, freq=[0.2, 40])
        )

        if "eeg1" in channel:
            # 保存信号数据
            with h5py.File(f'{save_path}/eeg1_x5.h5', 'w') as f:
                f.create_dataset(channel, data=main_ext_raw_data)

            # 保存均值
            with h5py.File(f'{save_path}/eeg1_mean5.h5', 'w') as f:
                f.create_dataset('mean', data=main_mean)

            # 保存标准差
            with h5py.File(f'{save_path}/eeg1_std5.h5', 'w') as f:
                f.create_dataset('std', data=main_std)

        if "eog" in channel:
            # 保存信号数据
            with h5py.File(f'{save_path}/eog1_x5.h5', 'w') as f:
                f.create_dataset(channel, data=main_ext_raw_data)

            # 保存均值
            with h5py.File(f'{save_path}/eog1_mean5.h5', 'w') as f:
                f.create_dataset('mean', data=main_mean)

            # 保存标准差
            with h5py.File(f'{save_path}/eog1_std5.h5', 'w') as f:
                f.create_dataset('std', data=main_std)

        # 保存标签数据（只保存一次）
        if not label_saved:
            with h5py.File(f'{save_path}/labels5.h5', 'w') as f:
                f.create_dataset('labels', data=main_labels)
            label_saved = True

if __name__ == '__main__':
    main()