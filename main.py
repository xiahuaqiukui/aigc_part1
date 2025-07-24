from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
from datetime import datetime
import shutil
import json

from data_preparations.new_single_epoch_Sleep_EDF_153 import Sleep_EDF_SC_signal_extract, Sleep_EDF_SC_signal_extract_WITHOUT_HY
from model.loader import SleepEDF_Seq_MultiChan_Dataset_Inference

import torch
from torchvision import transforms, datasets
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

app = FastAPI()

# 静态挂载：GET /sleep_results/... 即可拿到图片
app.mount("/sleep_results", StaticFiles(directory="sleep_results"), name="sleep_results")

MAX_SIZE = 10 * 1024 * 1024 * 1024  # 10 GB


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    # 大小校验,若文件过大则返回报错
    real_file = file.file
    real_file.seek(0, os.SEEK_END)
    if real_file.tell() > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    real_file.seek(0)

    # 获得并对原始数据进行预处理
    my_file = file.filename
    eeg_raw_data, eeg_sub_len, eeg_mean, eeg_std = (
        Sleep_EDF_SC_signal_extract_WITHOUT_HY(my_file, channel="eeg1", filter=True, freq=[0.2, 40])
    )
    eog_raw_data, _, eog_mean, eog_std = (
        Sleep_EDF_SC_signal_extract_WITHOUT_HY(my_file, channel="eog", filter=True, freq=[0.2, 40])
    )

    # 封装数据
    num_seq = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # 第一张图片, eeg_sample
    eeg_data_temp = eeg_data[0].squeeze()
    eog_data_temp = eog_data[0].squeeze()

    # print(eeg_data_temp.shape)

    t = np.arange(0, 30, 1 / 100)
    plt.figure(figsize=(10, 5))
    plt.plot(eeg_data_temp[0].squeeze())
    plt.plot(eog_data_temp[0].squeeze() + 5)
    plt.title(f"EEG & EOG表格")

    # ......
    # 保存第一张图



    # 生成输出目录
    patient_id = str(uuid.uuid4())[:8]
    ts = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = f"sleep_results/{patient_id}/{ts}"
    os.makedirs(out_dir, exist_ok=True)

    # 保存上传文件（示例）
    upload_path = os.path.join(out_dir, file.filename)
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(real_file, f)

    # TODO：后台处理，生成三张图
    for name in ["eeg_sample.jpg", "relationships.jpg", "interpretation.jpg"]:
        open(os.path.join(out_dir, name), "wb").close()

    # 返回 JSON（英文键名 + 强制 UTF-8）
    data = {
        "sleep_ratio": {"n1": 35, "n2": 25, "n3": 20, "w": 10, "rem": 10},
        "sleep_stages": ["n1", "n2", "n3", "w", "rem", "n2", "n3"],
        "eeg_signal": [1.2, 3.4, 5.6, 7.8],
        "eog_signal": [0.1, 0.2, 0.3],
        "sampling_rate_hz": 100,
        "eeg_preview_url": f"/sleep_results/{patient_id}/{ts}/eeg_sample.jpg",
        "relationship_img_url": f"/sleep_results/{patient_id}/{ts}/relationships.jpg",
        "interpretation_img_url": f"/sleep_results/{patient_id}/{ts}/interpretation.jpg"
    }
    '''
    中文意思
     return {
        "睡眠阶段占比": {"n1": 35, "n2": 25, "n3": 20, "w": 10, "rem": 10},
        "睡眠结构": ["n1", "n2", "n3", "w", "rem", "n2", "n3"],
        "脑电信号": [1.2, 3.4, 5.6, 7.8],
        "眼电信号": [0.1, 0.2, 0.3],
        "信号采样率": 100,
        "EEG-EOG预览地址": f"/sleep_results/{patient_id}/{ts}/eeg_sample.jpg",
        "跨模态权重图片地址": f"/sleep_results/{patient_id}/{ts}/relationships.jpg",
        "最终解释图片地址": f"/sleep_results/{patient_id}/{ts}/interpretation.jpg"
    }
    '''
    # JSONResponse 默认 charset=utf-8，保险起见显式指定
    return JSONResponse(content=data, media_type="application/json; charset=utf-8")



@app.get("/api/images/{patient_id}/{date}/{filename}")
def get_image(patient_id: str, date: str, filename: str):
    path = f"sleep_results/{patient_id}/{date}/{filename}"
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)