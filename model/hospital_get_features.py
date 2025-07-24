# -*- coding: utf-8 -*-

import os
import pandas as pd
from collections import defaultdict


def parse_annotations(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            onset = float(parts[0])
            duration = float(parts[1])
            desc = parts[2]
            data.append({'onset': onset, 'duration': duration, 'stage': desc})
    return data


def calculate_features(data):
    features = defaultdict(float)
    stages = [d['stage'] for d in data]
    onsets = [d['onset'] for d in data]
    durations = [d['duration'] for d in data]
    end_times = [d['onset'] + d['duration'] for d in data]
    trt = max(end_times) if data else 0  # 总记录时间

    # Sleep Latency (找到第一个非W阶段的onset)
    sleep_start = None
    for d in data:
        if 'W' not in d['stage']:
            sleep_start = d['onset']
            break
    features['Sleep latency'] = sleep_start if sleep_start is not None else 0

    # REM Latency (第一个R阶段onset - sleep_start)
    rem_start = None
    for d in data:
        if 'R' in d['stage']:
            rem_start = d['onset']
            break
    if rem_start is not None and sleep_start is not None:
        features['REM latency'] = rem_start - sleep_start
    else:
        features['REM latency'] = None

    # 累计各阶段时间
    stage_durations = defaultdict(float)
    for d in data:
        stage = d['stage'].split()[-1]  # 提取阶段标识（W,1,2,3,R）
        stage_durations[stage] += d['duration']

    tst = sum(v for k, v in stage_durations.items() if k != 'W')  # 总睡眠时间
    features['Sleep maintenance'] = tst / trt if trt > 0 else 0

    # 各阶段比例
    features['LS proportion'] = (stage_durations.get('1', 0) + stage_durations.get('2', 0)) / tst if tst > 0 else 0
    features['SWS proportion'] = stage_durations.get('3', 0) / tst if tst > 0 else 0
    features['REM proportion'] = stage_durations.get('R', 0) / tst if tst > 0 else 0

    # Arousal times (非W -> W的次数)
    arousal_count = 0
    prev_stage = None
    for d in data:
        current_stage = d['stage'].split()[-1]
        if prev_stage and prev_stage != 'W' and current_stage == 'W':
            arousal_count += 1
        prev_stage = current_stage
    features['Arousal times'] = arousal_count

    return features


def process_files(input_dir, output_file):
    results = []
    for filename in os.listdir(input_dir):
        if filename.endswith('_mne_Annotation.txt'):
            file_path = os.path.join(input_dir, filename)
            data = parse_annotations(file_path)
            features = calculate_features(data)
            features['File'] = filename
            features['isMDD'] = 1.
            results.append(features)

    df = pd.DataFrame(results)
    columns_order = ['File', 'Sleep latency', 'REM latency', 'LS proportion',
                     'SWS proportion', 'REM proportion', 'Sleep maintenance', 'Arousal times',"isMDD"]
    df = df[columns_order]
    df.to_excel(output_file, index=False)


# 使用示例

input_dir = r'./annotation'
output_file = r'./features/Hospital_sleep_features.xlsx'
process_files(input_dir, output_file)
print(f"特征已保存至 {output_file}")