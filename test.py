from model.fetch_seq import fetch_seq
import yaml

real_file = r"E:\hkk\项目_可解释睡眠分期\项目原始数据集\physionet-sleep-data\SC4001E0-PSG.edf"
with open("./model/model_config.yaml") as f:
    model_config = yaml.safe_load(f)
eeg, eog = fetch_seq(real_file, model_config)

print(eeg)
print(eog)