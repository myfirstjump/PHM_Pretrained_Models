import numpy as np
import torch
import importlib.metadata
import timesfm

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   
import matplotlib
matplotlib.use("Agg")                      

import matplotlib.pyplot as plt

def test_timesfm():
    # 環境資訊
    print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda,
          "| CUDA available:", torch.cuda.is_available())
    try:
        print("TimesFM (pkg metadata):", importlib.metadata.version("timesfm"))
    except importlib.metadata.PackageNotFoundError:
        print("TimesFM (pkg metadata): <unknown>")

    # 測試資料：sin 波
    t = np.linspace(0, 20, 120)
    noise = np.random.normal(0, 0.5, t.shape)
    # 加入雜訊
    t = t + noise
    y = np.sin(t)
    context = y[:100]
    horizon = 20
    inputs = [context]

    # 載入模型
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )

    cfg = timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True, #啟用一個額外的（~30M 參數）quantile head，讓模型能直接預測 連續分位數（支援到 1k horizon），輸出像看到的 (batch, horizon, num_quantiles)。
        force_flip_invariance=True, #  什麼時候開：訊號可能以 0 為中心正負擺動（例如振動、電流 AC 成分）→ 建議開。  
                                    #  什麼時候關：序列本質非負且負值沒有物理意義（例如銷量、件數）→ 影響不大，可關。
        infer_is_positive=False, # 對本質非負的目標（需求、流量、溫度 Kelvin 等）啟用非負性偏好/約束（常見做法是輸出後做下界截斷或以對數域/連續分布實現非負）。
                                 #  振動/電壓等可正可負訊號 → 設 False。
                                 #  功率、件數、壓力(≥0) → 設 True。
        fix_quantile_crossing=True, # 防止分位數交叉（例如 p10 > p50 的不合理情況）。常見作法是對每個時間步進行單調化（如 isotonic regression 或排序修正），做量化風險/區間可視化時 開。
    )
    model.compile(cfg)

    # 推論
    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon,
        inputs=inputs,
    )

    forecast = point_forecast[0]  # shape (horizon,)
    ground_truth = y[100:120]     # 真值

    # === 繪圖 ===
    plt.figure(figsize=(10, 5))
    # 原始資料
    plt.plot(range(len(y)), y, label="Full signal", color="gray", alpha=0.5)
    # context
    plt.plot(range(len(context)), context, label="Context (history)", color="blue")
    # 真值
    plt.plot(range(100, 120), ground_truth, label="Ground truth (future)", color="green")
    # 模型預測
    plt.plot(range(100, 120), forecast, label="TimesFM forecast", color="red", linestyle="--")

    # 若有 quantile 預測，加上 10% ~ 90% 區間
    if quantile_forecast is not None:
        p10 = quantile_forecast[0, :, 1]
        p90 = quantile_forecast[0, :, 9]
        plt.fill_between(range(100, 120), p10, p90, color="orange", alpha=0.2,
                         label="P10–P90 interval")

    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title("TimesFM 2.5 Forecast vs Ground Truth")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("timesfm_plot.png", dpi=150)

if __name__ == "__main__":
    test_timesfm()
