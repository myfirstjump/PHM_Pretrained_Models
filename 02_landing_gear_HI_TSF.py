
# 02_landing_gear_HI_TSF.py
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel, ConstantKernel
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.arima.model import ARIMA
# from pmdarima import auto_arima

# === 你剛寫的工具 ===
from py_modules.common_utils import (
    make_HI_series, read_rawdata, extract_features, feature_name,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ==== 路徑 ====
rootDir  = os.getcwd()
dataDir  = os.path.join(rootDir, "data", "LandingGear_relatives", "data")
modelDir = os.path.join(rootDir, "data", "LandingGear_relatives", "model")
predDir  = os.path.join(rootDir, "prediction", "F05")
os.makedirs(predDir, exist_ok=True)

# ==== 參數 ====
ava = "F05"
N_CONTEXT = 65
N_HORIZON = 16
# ==== 1) 由資料夾 → 特徵 → HI(CV) → MA50 ====
# 先處理前48架次（F05），再處理16架次（F05_prediction_gt），最後合併
folder_ctx = os.path.join(dataDir, "F05_custom")                 # 前65
folder_gt  = os.path.join(dataDir, "F05_prediction_gt")   # 後16
pipe_path  = os.path.join(modelDir, "hi_lr_pipeline_1141019_self_features.pkl")

print(f"[INFO] Processing context data from: {folder_ctx}")
'''
完成資料前處理，並進行rolling(MA5~MA50)
'''
ctx_df = make_HI_series(
    folder_path=folder_ctx,
    pipe_path=pipe_path,
    read_rawdata=read_rawdata,
    extract_features=extract_features,
    feature_name=feature_name,
)

print(f"[INFO] Processing ground-truth data from: {folder_gt}")
gt_df = make_HI_series(
    folder_path=folder_gt,
    pipe_path=pipe_path,
    read_rawdata=read_rawdata,
    extract_features=extract_features,
    feature_name=feature_name,
)

# 合併 (index=flight)，排序後成為完整 64 筆
hi_df = pd.concat([ctx_df, gt_df]).sort_index()
hi_df['MA05'] = hi_df['CV'].rolling(window=5, min_periods=1).mean()
hi_df['MA10'] = hi_df['CV'].rolling(window=10, min_periods=1).mean()
hi_df['MA20'] = hi_df['CV'].rolling(window=20, min_periods=1).mean()
hi_df['MA30'] = hi_df['CV'].rolling(window=30, min_periods=1).mean()
hi_df['MA40'] = hi_df['CV'].rolling(window=40, min_periods=1).mean()
hi_df['MA50'] = hi_df['CV'].rolling(window=50, min_periods=1).mean()

hi_df.to_csv(os.path.join(predDir, f"01_{ava}_HI_full.csv"))
print(f"[INFO] Saved full HI series ({len(hi_df)} flights) → {predDir}")

# ==== 2) 取前 48 架次當「模型輸入」 ====
ctx_df = hi_df.iloc[:N_CONTEXT].copy()
ctx_df[["MA05", "MA10", "MA20", "MA30", "MA40", "MA50"]].to_csv(os.path.join(predDir, f"02_{ava}_train_context_MA05-50.csv"))

# 未來 16 架次（**用連號外推**；若你的 flight 不是連號，可改成依實際 gt 的 flight 編號對齊）
last_flt = int(ctx_df.index[-1]) # F05-1398
future_index = np.arange(last_flt + 1, last_flt + 1 + N_HORIZON) # 1399-1415


for RND in ["MA05", "MA10", "MA20", "MA30", "MA40", "MA50"]:

    # ==== 3) 三種傳統模型：用「前 48 MA50」外推後 16 ====
    y_ctx = ctx_df[RND].values.astype(float)

    # (a) AR (自動挑階數)
    try:
        ar_order = ar_select_order(y_ctx, maxlag=10, glob=True, trend="ct")
        print(f'For {RND}, AR select order: {ar_order.ar_lags}')
        ar_res = ar_order.model.fit()
        ar_pred = ar_res.predict(start=len(y_ctx), end=len(y_ctx) + N_HORIZON - 1)
        ar_pred = np.asarray(ar_pred).ravel()
    except Exception as e:
        print("[AR] fallback:", e)
        ar_pred = np.full(N_HORIZON, np.nan)

    # (b) GPR
    k0 = WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))
    k1 = ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(20, 80))
    k2 = RBF(length_scale=1e2, length_scale_bounds=(1, 1e3))
    kernel = k0 + k1 + k2
    # ---
    # k0 = WhiteKernel(noise_level=0.05**2, noise_level_bounds=(1e-5, 0.2))
    # k1 = ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(20, 80))
    # k2 = RBF(length_scale=5.0, length_scale_bounds=(1e-1, 50))
    # kernel = k0 + k2
    # ---
    # kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(5.0, (1e-2, 50.0)) \
    #      + WhiteKernel(0.03**2, (1e-6, 0.1))

    X_train = np.arange(len(y_ctx)).reshape(-1, 1)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
    gpr.fit(X_train, y_ctx.reshape(-1, 1))
    X_future = np.arange(len(y_ctx), len(y_ctx) + N_HORIZON).reshape(-1, 1)
    gpr_pred, _ = gpr.predict(X_future, return_std=True)
    gpr_pred = gpr_pred.ravel()

    # (c) ARIMA：用 AIC 格點搜尋自動選 (p,d,q)
    # 搜尋範圍可以視資料長度調小一點，避免太慢
    best_aic = float("inf")
    best_order = None
    best_model = None

    # 這裡我假設你的序列已經是 MA 過的、也不算太長
    # 所以 d 只試 0 和 1 就好；p 試 0~5；q 試 0~3
    for p in range(0, 6):
        for d in range(0, 2):
            for q in range(0, 4):
                try:
                    tmp_model = ARIMA(y_ctx, order=(p, d, q)).fit()
                    tmp_aic = tmp_model.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (p, d, q)
                        best_model = tmp_model
                except Exception:
                    # 有些組合會發散或無法估計，跳過即可
                    continue

    if best_model is not None:
        print(f"[ARIMA-grid] {RND} best order (p,d,q) = {best_order}, AIC = {best_aic:.2f}")
        arima_pred = best_model.forecast(steps=N_HORIZON)
        arima_pred = np.asarray(arima_pred).ravel()
    else:
        # 萬一全部失敗，就給 NaN，避免整支程式掛掉
        print(f"[ARIMA-grid] {RND} no valid model found, fill NaN")
        arima_pred = np.full(N_HORIZON, np.nan)

    # ==== 4) 整理與輸出 ====
    pred_df = pd.DataFrame({
        "flight": future_index,
        "AR": ar_pred,
        "GPR": gpr_pred,
        "ARIMA": arima_pred,
    }).set_index("flight")

    # 存各模型各自檔（方便單獨調試）與整合檔
    # pred_df[["AR"]].to_csv(os.path.join(predDir, f"{ava}_AR_{RND}_pred16.csv"))
    # pred_df[["GPR"]].to_csv(os.path.join(predDir, f"{ava}_GPR_{RND}_pred16.csv"))
    # pred_df[["ARIMA"]].to_csv(os.path.join(predDir, f"{ava}_ARIMA_{RND}_pred16.csv"))
    pred_df.to_csv(os.path.join(predDir, f"03_{ava}_traditional_{RND}_pred{N_HORIZON}_all.csv"))

    # 也保留「輸入＋預測」一起的 64 點（前 48 + 後 16）供畫圖
    plot_64 = pd.concat(
        [ctx_df[[RND]], pred_df.rename(columns={"AR":f"{RND}_AR","GPR":f"{RND}_GPR","ARIMA":f"{RND}_ARIMA"})],
        axis="columns"
    )
    plot_64.to_csv(os.path.join(predDir, f"04_{ava}_{RND}_context{N_CONTEXT}_and_pred{N_HORIZON}_for_plot.csv"))





## ==== 5) 視覺化（全長 64 點，只畫 MA50 與三條預測）====
    plt.figure(figsize=(8, 5))
    # context（實際 MA50）
    plt.plot(ctx_df.index.values, ctx_df[RND].values, label=f"Context", linewidth=2)

    # 三條外推
    plt.plot(future_index, ar_pred,    "--", label="AR pred",    linewidth=2)
    plt.plot(future_index, gpr_pred,   "--", label="GPR pred",   linewidth=2)
    plt.plot(future_index, arima_pred, "--", label="ARIMA pred", linewidth=2)

    plt.title(f"Flight00 | context={N_CONTEXT} → forecast {N_HORIZON} (AR/GPR/ARIMA)")
    plt.xlabel("Flight")
    plt.ylabel(f"HI")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(predDir, f"05_{ava}_{RND}_context{N_CONTEXT}_pred{N_HORIZON}.png"), dpi=160)
    plt.show()




# === NEW: 畫出「尚未MA」的 context 純散點圖 (使用 CV) ===
try:
    ctx_x  = ctx_df.index.values
    ctx_cv = ctx_df["CV"].astype(float).values

    plt.figure(figsize=(9,4.5))
    plt.scatter(ctx_x, ctx_cv, s=26, alpha=0.9)
    plt.title(f"Flight00 context HI — first {N_CONTEXT} flights")
    plt.xlabel("Flight")
    plt.ylabel("HI")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(predDir, f"06_{ava}_ctx_CV_scatter_{N_CONTEXT}.png")
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[PLOT] Saved scatter (pre-MA CV): {out_png}")
except Exception as e:
    print("[PLOT] Scatter plot failed:", e)
