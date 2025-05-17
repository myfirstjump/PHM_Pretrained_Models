# -*- coding: utf-8 -*-
import timesfm

# 初始化模型
tfm = timesfm.TimesFm(
    context_len=512,
    horizon_len=128,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend="pytorch",
)

# 從 Hugging Face 下載並載入模型權重
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m-pytorch")

import numpy as np

# 準備輸入數據
forecast_input = [np.sin(np.linspace(0, 20, 100))]
frequency_input = [0]  # 0: 高頻率（例如每日數據）

# 執行預測
point_forecast, _ = tfm.forecast(forecast_input, freq=frequency_input)

print(point_forecast)