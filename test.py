# test.py
import requests
import os
import uuid

# 1. 本地文件路径
FILE_PATH = r"E:\mne_data\physionet-sleep-data\SC4001E0-PSG.edf"
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(FILE_PATH)

# 2. FastAPI 地址
BASE_URL = "http://localhost:8000"
UPLOAD_URL = f"{BASE_URL}/api/upload"

# 3. 上传
with open(FILE_PATH, "rb") as f:
    resp = requests.post(
        UPLOAD_URL,
        files={"file": (os.path.basename(FILE_PATH), f, "application/octet-stream")}
    )
    resp.raise_for_status()

data = resp.json()
print("返回 JSON：", data)

# 4. 下载三张图片
for key, url in [
    ("eeg_preview", data["eeg_preview_url"]),
    ("relationship", data["relationship_img_url"]),
    ("interpretation", data["interpretation_img_url"]),
]:
    img_resp = requests.get(f"{BASE_URL}{url}")
    img_resp.raise_for_status()
    out_name = f"{key}_{uuid.uuid4().hex[:6]}.jpg"
    with open(out_name, "wb") as img_file:
        img_file.write(img_resp.content)
    print(f"已保存 {out_name}")