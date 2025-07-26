from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
from datetime import datetime
import shutil
import json
import yaml

from data_preparations.new_single_epoch_Sleep_EDF_153 import Sleep_EDF_SC_signal_extract, Sleep_EDF_SC_signal_extract_WITHOUT_HY
from model.loader import SleepEDF_Seq_MultiChan_Dataset_Inference

from model.model_run_util import model_run
from model.fetch_seq import fetch_seq
import torch
from torchvision import transforms, datasets
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from model.sequence_cmt import (Seq_Cross_Transformer_Network, Epoch_Cross_Transformer, Intra_modal_atten,
                                Window_Embedding, PositionalEncoding, Cross_modal_atten,Feed_forward)




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

    # 生成输出目录
    patient_id = str(uuid.uuid4())[:8]
    ts = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = f"sleep_results/{patient_id}/{ts}"
    os.makedirs(out_dir, exist_ok=True)

    # 保存上传文件
    upload_path = os.path.join(out_dir, file.filename)
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(real_file, f)

    # 加载配置文件
    with open("model/model_config.yaml") as f:
        model_config = yaml.safe_load(f)

    # 调用模型函数model_run：包括数据预处理以及结果保存,返回睡眠阶段占比和睡眠结构
    # 这里传入的参数file_path是文件的地址
    result_dict, pred_stages = model_run(upload_path, out_dir, model_config)
    eeg_signal, eog_signal = fetch_seq(upload_path, model_config)


    # 返回 JSON（英文键名 + 强制 UTF-8）
    data = {
        "sleep_ratio": result_dict,
        "sleep_stages": pred_stages,
        # "eeg_signal": [1.2, 3.4, 5.6, 7.8],
        # "eog_signal": [0.1, 0.2, 0.3],
        "eeg_signal": eeg_signal,
        "eog_signal": eog_signal,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)