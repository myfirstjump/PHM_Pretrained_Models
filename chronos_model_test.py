# pip install chronos-forecasting matplotlib numpy
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import ChronosPipeline

# ==== 1) 造一段單變量時間序列 ====
np.random.seed(42)
n_total = 200
t = np.arange(n_total)
series = (
    2.0 * np.sin(2*np.pi*t/24)      # 週期
    + 0.02 * t                      # 趨勢
    + 0.5 * np.random.randn(n_total)  # 雜訊
).astype(np.float32)

context_len = 150
horizon = 24
context_np = series[:context_len]
future_truth = series[context_len:context_len+horizon]  # 只是用來畫圖對照

# ==== 2) 載入 Chronos ====
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 選 dtype：GPU 優先 bfloat16，其次 float16；CPU 用 float32
if device.startswith("cuda"):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    dtype = torch.float32

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map=device,
    dtype=dtype,  # << 用 dtype 取代 torch_dtype
)

# ==== 3) 準備 context：必須是 torch.Tensor 且形狀 (batch, context_len) ====
# (1) 轉 torch；(2) 新增 batch 維度；(3) Chronos 會自己搬到裝置上，這裡不強制 .to(device)
context = torch.tensor(context_np, dtype=torch.float32).unsqueeze(0)  # shape: (1, context_len)

# ==== 4) 抽樣預測 ====





# 取得抽樣
num_samples = 200
samples_t = pipeline.predict(
    context=context,                # torch.Tensor, shape: (1, context_len)
    prediction_length=horizon,
    num_samples=num_samples,
)  # -> shape: (S, B, H)

# --- 形狀整理與檢查 ---
assert isinstance(samples_t, torch.Tensor)
print("raw samples shape:", tuple(samples_t.shape))  # 期待 (S, B, H)

# 壓掉 batch 維度，得到 (S, H)
if samples_t.ndim == 3:
    # S, B, H
    samples_t = samples_t[:, 0, :]
elif samples_t.ndim == 2:
    # 已是 (S, H)
    pass
else:
    raise ValueError(f"Unexpected samples shape: {samples_t.shape}")

print("squeezed samples shape:", tuple(samples_t.shape))  # 期待 (S, H) = (200, 24)

# --- 計算分位數（沿著樣本維度 dim=0）---
# 這一步的結果應該是 (H,) = (24,)
p50 = torch.quantile(samples_t, 0.5, dim=0).cpu().numpy()
p10 = torch.quantile(samples_t, 0.10, dim=0).cpu().numpy()
p90 = torch.quantile(samples_t, 0.90, dim=0).cpu().numpy()
print("p50 shape:", p50.shape)  # 期待 (24,)

# --- 作圖 ---
x_future = t[context_len:context_len + horizon]  # (24,)
assert p50.shape == x_future.shape
assert p10.shape == x_future.shape
assert p90.shape == x_future.shape




# ==== 5) 視覺化 ====
# x 軸（未來）
x_future = t[context_len:context_len + horizon]              # (H,)
# x 軸（歷史）
x_hist   = t[:context_len]                                   # (L,)

# 安全檢查：確保分位數與 x_future 對齊
assert p50.shape == x_future.shape == p10.shape == p90.shape, \
    f"shape mismatch: p50={p50.shape}, p10={p10.shape}, p90={p90.shape}, x_future={x_future.shape}"

fig, ax = plt.subplots(figsize=(10, 4))

# 歷史序列（若不需要可註解掉）
ax.plot(x_hist, context_np, label="History", linewidth=2)

# 預測中位數 + 信賴帶
ax.plot(x_future, p50, label="Forecast (P50)", linewidth=2)
ax.fill_between(x_future, p10, p90, alpha=0.25, label="P10–P90")

# 若你有未來真值（在玩具資料裡有），可一併畫出來
if 'future_truth' in globals() and future_truth.shape == x_future.shape:
    ax.plot(x_future, future_truth, linestyle="--", label="Ground Truth")

ax.set_title(f"Univariate Forecast with Chronos (T5 Small) | device={device}, dtype={dtype}")
ax.set_xlabel("time")
ax.set_ylabel("value")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()