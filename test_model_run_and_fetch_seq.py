import yaml
import uuid
from datetime import datetime
from model.model_run_util import model_run
from model.fetch_seq import fetch_seq

from model.sequence_cmt import (Seq_Cross_Transformer_Network, Epoch_Cross_Transformer, Intra_modal_atten,
                                Window_Embedding, PositionalEncoding, Cross_modal_atten,Feed_forward)


upload_path = r"D:\项目原始数据集\SC4001E0-PSG.edf"

patient_id = str(uuid.uuid4())[:8]
ts = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
out_dir = f"sleep_results/{patient_id}/{ts}"

with open("model/model_config.yaml") as f:
    model_config = yaml.safe_load(f)

result_dict, pred_stages = model_run(upload_path, out_dir, model_config)
eeg_signal, eog_signal = fetch_seq(upload_path, model_config)
