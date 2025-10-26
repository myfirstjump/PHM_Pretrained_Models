# import os
# import warnings

# import joblib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from statsmodels.tsa.ar_model import ar_select_order
# from statsmodels.tsa.arima.model import ARIMA

# from py_modules.Step1_Data_Preprocessing import read_rawdata
# from py_modules.Step2_Feature_Extraction import extract_features, feature_name
# from py_modules.utils import make_HI_series




# rootDir     = os.getcwd()               # root directory
# dataDir     = f'{rootDir}\\data\\LandingGear_relatives\\data'        # data directory
# csvDir      = f'{rootDir}\\data\\LandingGear_relatives\\csv'         # csv directory
# featuresDir = f'{rootDir}\\data\\LandingGear_relatives\\myfeature'   # features directory
# modelDir    = f'{rootDir}\\data\\LandingGear_relatives\\model'       # models directory

# ### 載入資料前處理pipeline
# pipe: Pipeline = joblib.load(f'{modelDir}\\hi_lr_pipeline_1141019_self_features.pkl')
# ### Find how many fly we have:

# ava_names = list(set(csv[:3] for csv in os.listdir(f'{dataDir}\\F05')))
# print('ava_names:', ava_names)
# ### Read F05 rawdata to csv dataset
# ## create empty lists for dataframe concat operation
# flight_TO_Y_dfs = []
# flight_TO_Z_dfs = []
# flight_LD_Y_dfs = []
# flight_LD_Z_dfs = []
# flight_num_list = []

# ## read rawdata by read_rawdata() and append the ouput series to the lists
# '''
# 必須要用F05的資料，才有足夠多的數據點進行TSF任務
# '''
# ava = 'F05'
# for csv in os.listdir(f'{dataDir}\\F05'):

#     flight_num_list.append(int(csv[-9:-4]))
#     flight_TO_Y_df, flight_TO_Z_df, flight_LD_Y_df, flight_LD_Z_df = read_rawdata(f'{dataDir}\\F05\\{csv}')
#     flight_TO_Y_dfs.append(flight_TO_Y_df)
#     flight_TO_Z_dfs.append(flight_TO_Z_df)
#     flight_LD_Y_dfs.append(flight_LD_Y_df)
#     flight_LD_Z_dfs.append(flight_LD_Z_df)
# ## concat all the series in the lists to dateframes
# F05_TO_Y_Dataset = pd.concat(flight_TO_Y_dfs, axis='columns')
# F05_TO_Z_Dataset = pd.concat(flight_TO_Z_dfs, axis='columns')
# F05_LD_Y_Dataset = pd.concat(flight_LD_Y_dfs, axis='columns')
# F05_LD_Z_Dataset = pd.concat(flight_LD_Z_dfs, axis='columns')

# ## save the dataframes to csv files
# F05_TO_Y_Dataset.to_csv(f'{csvDir}\\each_ava\\{ava}_TO_Y_Dataset.csv', index=False)
# F05_TO_Z_Dataset.to_csv(f'{csvDir}\\each_ava\\{ava}_TO_Z_Dataset.csv', index=False)
# F05_LD_Y_Dataset.to_csv(f'{csvDir}\\each_ava\\{ava}_LD_Y_Dataset.csv', index=False)
# F05_LD_Z_Dataset.to_csv(f'{csvDir}\\each_ava\\{ava}_LD_Z_Dataset.csv', index=False)

# ### Extract features from F05 dataset
# F05_dataset_list = [f'{ava}_TO_Y_Dataset.csv', f'{ava}_TO_Z_Dataset.csv', f'{ava}_LD_Y_Dataset.csv', f'{ava}_LD_Z_Dataset.csv']
# tmp_dfs = []
# for dataset in F05_dataset_list:
#     tmp_features = extract_features(f'{csvDir}\\each_ava\\{dataset}')
#     tmp_dfs.append(tmp_features)

# F05AllFeatures = pd.concat(tmp_dfs, axis='columns')
# F05AllFeatures.columns = feature_name
# # F05AllFeatures.to_csv(f'{featuresDir}\\F05AllFeatures.csv', index=False)

# ### Caculate CV 
# ## Normalization 
# # sc:StandardScaler = joblib.load(f'{modelDir}\\sc.pkl')  # load normalize model
# # # sc:StandardScaler = joblib.load(f'{modelDir}\\sc_allfeatures1141017.pkl')  # load normalize model
# # F05Features_norm = sc.transform(F05AllFeatures)

# # ### Apply PCA
# # pca = joblib.load(f'{modelDir}\\pca_onlyTrain1141019.pkl')
# # # pca = joblib.load(f'{modelDir}\\pca_allfeatures1141017.pkl')
# # F05Features_norm = pca.transform(F05Features_norm)
# # print('================PCA=================START', )
# # print(F05Features_norm)
# # print('================PCA=================END', )
# # ### Load Logistics Regresion as Health Indicator
# # LR_model:LogisticRegression = joblib.load(f'{modelDir}\\classifier_0.pkl')
# # # LR_model:LogisticRegression = joblib.load(f'{modelDir}\\all_data_LR_Health_Indicator_Curve1141017.pkl')
# # HI = LR_model.predict_proba(F05Features_norm)  # load LogisticRegression model

# # 1) 取得 F05 特徵，確保欄位順序與訓練時一致
# #    (feature_name 要與訓練資料的 columns 完全相同 & 同順序)
# # pipe.fit(F05AllFeatures)

# # 2) 直接用 pipeline 預測機率
# proba = pipe.predict_proba(F05AllFeatures.values) # predict_proba得到兩個欄位資料(LR預測0的機率, LR預測1的機率) -->   [:,0] 對應 class_ == 0; [:,1] 對應 class_ == 1
# print('================proba===========START', )
# print(proba)
# print('================proba===========END', )


# # # 3) 取「健康(0)」的機率欄
# # i0 = np.where(pipe.named_steps['lr'].classes_ == 0)[0][0]
# # HI = proba[:, i0]
# healthy_idx = np.where(pipe.named_steps['lr'].classes_ == 0)[0][0]
# faulty_idx  = np.where(pipe.named_steps['lr'].classes_ == 1)[0][0]
# HI = proba[:, healthy_idx]


# print('================LR predict prob===========START', )
# print(HI)
# print('================LR predict prob===========END', )
# # HI = HI[:, 0]
# ## Data alignment
# HI_df = pd.DataFrame()
# HI_df['flight'] = flight_num_list
# HI_df['CV'] = HI
# HI_df = HI_df.set_index('flight')
# idx = pd.RangeIndex(start=HI_df.index.values.min(), stop=HI_df.index.values.max()+1, name='flight')
# HI_df = HI_df.reindex(idx)  # align data with actual flight order


# # ## Moving average MA50
# HI_df['MA50'] = HI_df['CV'].rolling(window=50, min_periods=1).mean()

# print('================CV DF=================START', )
# print(HI_df)

# print('================CV DF=================END', )


# os.makedirs(f"{csvDir}", exist_ok=True)
# HI_df.to_csv(f"{csvDir}\\F05_MA50_series.csv", index=True)
# print("Saved:", f"{csvDir}\\F05_MA50_series.csv", "| length:", len(HI_df))


# ### Caculate RUL

# H = 600
# ## AR
# ### https://www.statsmodels.org/devel/generated/statsmodels.tsa.ar_model.ar_select_order.html
# AR_mod = ar_select_order(HI_df['MA50'].values, 10, glob=True, trend='ct')   # statsmodels版本要0.12以上,trend=‘ct’ - Constant and time trend.
# AR_res = AR_mod.model.fit()
# AR_pred = AR_res.predict(start=0, end=H-1) ### Python 3.9版本之後，請將start=1改成start=0

# ## GPR
# k0 = WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))
# k1 = ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45))
# k2 = RBF(length_scale=1e2, length_scale_bounds=(1, 1e3))
# kernel = k0 + k1 + k2

# X_train_0 = HI_df['MA50'].shape
# X_train = np.atleast_2d(np.array(list(range(X_train_0[0])))).T
# Y_train = np.atleast_2d(HI_df['MA50'].values).T
# X = np.atleast_2d(list(range(600))).T

# gpr_mod = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
# gpr_res = gpr_mod.fit(X_train, Y_train)
# X = np.arange(H).reshape(-1, 1)
# GPR_pred, y_std = gpr_res.predict(X, return_std=True)

# # ARIMA
# ARIMA_mod = ARIMA(HI_df['MA50'].values, order=(2, 1, 0), trend='t')
# ARIMA_res = ARIMA_mod.fit(method='innovations_mle')
# ARIMA_pred = ARIMA_res.predict(1, H, typ='levels')

# nan_num = 600-len(AR_pred)
# AR_pred = nan_num * [np.nan] + AR_pred.tolist()

# # ---- build pre_df with H rows ----
# pre_index = np.arange(flight_num_list[0], flight_num_list[0] + H)
# pre_df = pd.DataFrame({
#     'flight': pre_index,
#     'AR': np.asarray(AR_pred).ravel()[:H],
#     'GPR': np.asarray(GPR_pred).ravel()[:H],
#     'ARIMA': np.asarray(ARIMA_pred).ravel()[:H],
# }).set_index('flight')





# # === NEW: save inputs & predictions for evaluation ===
# from pathlib import Path
# import json

# predictionDir = Path(rootDir) / 'data' / 'LandingGear_relatives' / 'prediction'
# predictionDir.mkdir(parents=True, exist_ok=True)

# # 1) 存 HI 與 MA50（做為所有模型的共同輸入和 GT 來源）
# hi_path = predictionDir / f'{ava}_HI_MA50.csv'
# HI_df.to_csv(hi_path, index=True)   # index=flight；含欄位 ['CV','MA50']

# # 2) 存三個模型的「輸入紀錄」與「超參數/擬合資訊」
# # --- AR ---
# ar_meta = {
#     "model": "AR",
#     "trend": getattr(AR_mod, "trend", "ct"),
#     "selected_lags": getattr(AR_mod, "ar_lags", None),
#     "order_max": 10,
#     "H": H,
#     "note": "AR_select_order(glob=True, trend='ct') on MA50"
# }
# (pd.DataFrame({"t": np.arange(len(HI_df['MA50'])),
#                "y": HI_df['MA50'].values})
#  .to_csv(predictionDir / f'{ava}_AR_input.csv', index=False))
# with open(predictionDir / f'{ava}_AR_meta.json', 'w', encoding='utf-8') as f:
#     json.dump(ar_meta, f, ensure_ascii=False, indent=2)

# # --- GPR ---
# gpr_meta = {
#     "model": "GPR",
#     "kernel_repr": str(gpr_res.kernel_),
#     "H": H,
#     "X_train_shape": list(X_train.shape),
#     "note": "WhiteKernel + ExpSineSquared(period≈40) + RBF on MA50"
# }
# (pd.DataFrame({"t": X_train.ravel(), "y": Y_train.ravel()})
#  .to_csv(predictionDir / f'{ava}_GPR_input.csv', index=False))
# with open(predictionDir / f'{ava}_GPR_meta.json', 'w', encoding='utf-8') as f:
#     json.dump(gpr_meta, f, ensure_ascii=False, indent=2)

# # --- ARIMA ---
# arima_meta = {
#     "model": "ARIMA",
#     "order": (2, 1, 0),
#     "trend": "t",
#     "fit_method": "innovations_mle",
#     "H": H,
#     "note": "ARIMA on MA50 (levels)"
# }
# (pd.DataFrame({"t": np.arange(len(HI_df['MA50'])),
#                "y": HI_df['MA50'].values})
#  .to_csv(predictionDir / f'{ava}_ARIMA_input.csv', index=False))
# with open(predictionDir / f'{ava}_ARIMA_meta.json', 'w', encoding='utf-8') as f:
#     json.dump(arima_meta, f, ensure_ascii=False, indent=2)

# # 3) 存「三個模型的預測」與「預測起點」
# pre_df_path = predictionDir / f'{ava}_classic_preds.csv'
# pre_df.to_csv(pre_df_path, index=True)  # index=flight；欄位 ['AR','GPR','ARIMA']

# with open(predictionDir / f'{ava}_forecast_anchor.json', 'w', encoding='utf-8') as f:
#     json.dump({
#         "forecast_start_flight": int(HI_df.index.values.max()),  # 預測線開始畫的 x
#         "first_flight_in_series": int(flight_num_list[0]),
#         "H": int(H)
#     }, f, ensure_ascii=False, indent=2)

# print(f'[Saved] inputs & predictions to: {predictionDir}')




# # ### Data visualization
# # ## Plot CV
# # plt.figure()
# # plt.plot(HI_df['CV'])
# # plt.title('F05 Original CV')
# # plt.xlabel('Flight')
# # plt.ylabel('CV value')
# # plt.xlim(flight_num_list[0], flight_num_list[0]+600)
# # plt.ylim(0, 1)
# # plt.grid()
# # plt.show()

# # ## Plot MA50 CV
# # plt.figure()
# # plt.plot(HI_df['MA50'])
# # plt.title('F05 MA50 CV')
# # plt.xlabel('Flight')
# # plt.ylabel('CV value')
# # plt.xlim(flight_num_list[0], flight_num_list[0]+600)
# # plt.ylim(0, 1)
# # plt.grid()
# # plt.show()

# # ## Plot RUL(AR, GPR, ARIMA)
# # plt.figure()
# # plt.plot(pre_df.loc[HI_df.index.values.max():].reset_index()['flight'].values, pre_df['AR'].loc[HI_df.index.values.max():].values, linestyle="--", c='red', label="AR prediction")
# # plt.plot(pre_df.loc[HI_df.index.values.max():].reset_index()['flight'].values, pre_df['GPR'].loc[HI_df.index.values.max():].values, linestyle="--", c='blue', label="GPR prediction")
# # plt.plot(pre_df.loc[HI_df.index.values.max():].reset_index()['flight'].values, pre_df['ARIMA'].loc[HI_df.index.values.max():].values, linestyle="--", c='orange', label="ARIMA prediction")
# # plt.plot(HI_df['MA50'].loc[flight_num_list[0]:HI_df.index.values.max()])
# # plt.title('F05 RUL')
# # plt.xlabel('Flight')
# # plt.ylabel('CV value')
# # plt.xlim(flight_num_list[0], flight_num_list[0]+600)
# # plt.ylim(0, 1)
# # plt.grid()
# # plt.legend()
# # plt.show()









# 02_landing_gear_HI_TSF.py
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.arima.model import ARIMA

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
N_CONTEXT = 42
N_HORIZON = 6
# ==== 1) 由資料夾 → 特徵 → HI(CV) → MA50 ====
# 先處理前48架次（F05），再處理16架次（F05_prediction_gt），最後合併
folder_ctx = os.path.join(dataDir, "F05")                 # 前48
folder_gt  = os.path.join(dataDir, "F05_prediction_gt")   # 後16
pipe_path  = os.path.join(modelDir, "hi_lr_pipeline_1141019_self_features.pkl")

print(f"[INFO] Processing context data from: {folder_ctx}")
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
hi_df.to_csv(os.path.join(predDir, f"{ava}_HI_full.csv"))
print(f"[INFO] Saved full HI series ({len(hi_df)} flights) → {predDir}")

# 依照 flight 排序並保存完整序列（方便後續評估與畫圖）
hi_df = hi_df.sort_index()
hi_df.to_csv(os.path.join(predDir, f"{ava}_HI_full.csv"))   # flight, CV, MA50

# ==== 2) 取前 48 架次當「模型輸入」 ====
ctx_df = hi_df.iloc[:N_CONTEXT].copy()
ctx_df[["MA20", "MA30", "MA40", "MA50"]].to_csv(os.path.join(predDir, f"{ava}_train_context_MA20-50.csv"))

# 未來 16 架次（**用連號外推**；若你的 flight 不是連號，可改成依實際 gt 的 flight 編號對齊）
last_flt = int(ctx_df.index[-1]) # F05-1398
future_index = np.arange(last_flt + 1, last_flt + 1 + N_HORIZON) # 1399-1415


for RND in ["MA20", "MA30", "MA40", "MA50"]:

    # ==== 3) 三種傳統模型：用「前 48 MA50」外推後 16 ====
    y_ctx = ctx_df[RND].values.astype(float)

    # (a) AR (自動挑階數)
    try:
        ar_order = ar_select_order(y_ctx, maxlag=10, glob=True, trend="ct")
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

    X_train = np.arange(len(y_ctx)).reshape(-1, 1)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=False)
    gpr.fit(X_train, y_ctx.reshape(-1, 1))
    X_future = np.arange(len(y_ctx), len(y_ctx) + N_HORIZON).reshape(-1, 1)
    gpr_pred, _ = gpr.predict(X_future, return_std=True)
    gpr_pred = gpr_pred.ravel()

    # (c) ARIMA(2,1,0)（可再調整）
    try:
        arima = ARIMA(y_ctx, order=(2, 1, 0), trend='t')
        arima_res = arima.fit(method='innovations_mle')
        arima_pred = arima_res.forecast(steps=N_HORIZON)
        arima_pred = np.asarray(arima_pred).ravel()
    except Exception as e:
        print("[ARIMA] fallback:", e)
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
    pred_df.to_csv(os.path.join(predDir, f"{ava}_traditional_{RND}_pred16_all.csv"))

    # # 也保留「輸入＋預測」一起的 64 點（前 48 + 後 16）供畫圖
    # plot_64 = pd.concat(
    #     [ctx_df[["MA50"]], pred_df.rename(columns={"AR":"MA50_AR","GPR":"MA50_GPR","ARIMA":"MA50_ARIMA"})],
    #     axis="columns"
    # )
    # plot_64.to_csv(os.path.join(predDir, f"{ava}_context48_and_pred16_for_plot.csv"))

# # ==== 5) 視覺化（全長 64 點，只畫 MA50 與三條預測）====
# plt.figure(figsize=(8, 5))
# # context（實際 MA50）
# plt.plot(ctx_df.index.values, ctx_df["MA50"].values, label="MA50 (first 48)", linewidth=2)

# # 三條外推
# plt.plot(future_index, ar_pred,    "--", label="AR pred",    linewidth=2)
# plt.plot(future_index, gpr_pred,   "--", label="GPR pred",   linewidth=2)
# plt.plot(future_index, arima_pred, "--", label="ARIMA pred", linewidth=2)

# plt.title(f"{ava} | MA50 context=48 → forecast 16 (AR/GPR/ARIMA)")
# plt.xlabel("Flight")
# plt.ylabel("CV (MA50)")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(predDir, f"{ava}_context48_pred16.png"), dpi=160)
# plt.show()
