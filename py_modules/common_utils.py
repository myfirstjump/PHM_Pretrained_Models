# --- common utils: HI + MA50 （單邊） ---
import os, json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import kurtosis, skew


def causal_ma(series: pd.Series, window=50):
    return series.rolling(window=window, min_periods=1).mean()

def flights_from_folder(folder_path):
    """讀資料夾內所有 flight 檔名 → 依檔名排序回傳 [(flight_num, file_path), ...]"""
    items = []
    for name in os.listdir(folder_path):
        if not name.lower().endswith(('.csv', '.txt')):
            continue
        # 假設檔名末 5 位為 flight 編號，如 xxxx_01428.csv → 01428
        digits = ''.join(ch for ch in name if ch.isdigit())
        flight = int(digits[-5:]) if len(digits) >= 5 else int(digits)
        items.append((flight, os.path.join(folder_path, name)))
    items.sort(key=lambda x: x[0])
    return items

def make_HI_series(folder_path, pipe_path, read_rawdata, extract_features, feature_name):
    import joblib
    pipe = joblib.load(pipe_path)

    flights_files = flights_from_folder(folder_path)

    rows = []
    idx  = []
    for flt, fp in flights_files:
        TO_Y, TO_Z, LD_Y, LD_Z = read_rawdata(fp)

        feats = []
        for seg in [TO_Y, TO_Z, LD_Y, LD_Z]:
            # seg 是 Series；先包成單欄 DataFrame，欄名用原本的 seg.name（若沒有就補一個）
            colname = seg.name if getattr(seg, "name", None) else f"F{flt:05d}"
            seg_df = pd.DataFrame({colname: seg})
            # 這樣 extract_features 才會把「每欄=一個 flight」當作輸入
            f10 = extract_features(seg_df).reset_index(drop=True)  # (1,10)
            feats.append(f10)

        # 四段各10維 → 橫向拼成 (1,40)，並改成你的40個特徵名稱
        df_f = pd.concat(feats, axis="columns")
        df_f.columns = feature_name

        rows.append(df_f)
        idx.append(flt)

    feats_all = pd.concat(rows, axis="rows").reset_index(drop=True)

    proba = pipe.predict_proba(feats_all.values)
    healthy_idx = int(np.where(pipe.named_steps["lr"].classes_ == 0)[0][0])
    CV = proba[:, healthy_idx]

    ser = pd.Series(CV, index=pd.Index(idx, name="flight")).sort_index()
    MA05 = causal_ma(ser, window=5)
    MA10 = causal_ma(ser, window=10)
    MA20 = causal_ma(ser, window=20)
    MA30 = causal_ma(ser, window=30)
    MA40 = causal_ma(ser, window=40)
    MA50 = causal_ma(ser, window=50)
    out = pd.DataFrame({"CV": ser.values, "MA05": MA05.values, "MA10": MA10.values, "MA20": MA20.values, "MA30": MA30.values, "MA40": MA40.values, "MA50": MA50.values}, index=ser.index)
    return out



### Define directories
rootDir            = os.getcwd()                            # root directory
trainingHealthyDir = f'{rootDir}\\data\\training\\Healthy'  # training healthy data directory
trainingFaultyDir  = f'{rootDir}\\data\\training\\Faulty'   # training faulty data directory
testingDir         = f'{rootDir}\\data\\testing'            # testing data directory
csvDir             = f'{rootDir}\\csv'                      # final output csv directory

'''
PEP 498 – Literal String Interpolation - f-string
'''

### Read rawdata
def read_rawdata(csv_path: str) -> pd.DataFrame:
    '''function of reading csv rawdata to dataframe'''
    
    flight_num = int(csv_path[-9:-4])                    # get flight number from filename
    plane_num  = str(csv_path.split('\\')[-1][:-10])    # get plane number from filename

    try:
        TO_index = pd.read_csv(csv_path, header=None, usecols=[0], nrows=1).iloc[0, 0]  # get TO data index
        LD_index = pd.read_csv(csv_path, header=None, usecols=[1], nrows=1).iloc[0, 0]  # get LD data index
        # 友達版本
        # Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
        # TO_index = pd.read_csv(csv_path, header=None, usecols=[0], nrows=1, engine='python').index[0][0]  # get takeoff data index
        # LD_index = pd.read_csv(csv_path, header=None, usecols=[1], nrows=1, engine='python').index[0][1] # get landing data index
    except Exception as e:
        raise Exception(f'{plane_num}-{flight_num:05d}: error obtaining T/O & L/D data index, {e}') 

    try:
        flight_df = pd.read_csv(csv_path, header=1)     # read rawdata csv
    except Exception as e:
        raise Exception(f'{plane_num}-{flight_num:05d}: error reading rawdata, {e}') 
    
    flight_TO_df = flight_df[:TO_index+1]   # get takeoff data
    flight_LD_df = flight_df[-LD_index:]    # get landing data

    try:
        flight_TO_df = flight_TO_df[flight_TO_df.C > 48]    # select takeoff data (airspeed > 48knots)
        flight_LD_df = flight_LD_df[flight_LD_df.C > 48]    # select landing data (airspeed > 48knots)
    except Exception as e:
        raise Exception(f'{plane_num}-{flight_num:05d}: error selecting data by airspeed, {e}') 

    # reset index, letting index start from 0
    flight_TO_Y_df = flight_TO_df.Y.reset_index(drop=True)
    flight_TO_Z_df = flight_TO_df.Z.reset_index(drop=True)
    flight_LD_Y_df = flight_LD_df.Y.reset_index(drop=True)
    flight_LD_Z_df = flight_LD_df.Z.reset_index(drop=True)

    # set name of the series
    flight_TO_Y_df.name = f'{plane_num}-{flight_num:05d}'
    flight_TO_Z_df.name = f'{plane_num}-{flight_num:05d}'
    flight_LD_Y_df.name = f'{plane_num}-{flight_num:05d}'
    flight_LD_Z_df.name = f'{plane_num}-{flight_num:05d}'
    
    return flight_TO_Y_df, flight_TO_Z_df, flight_LD_Y_df, flight_LD_Z_df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 10 statistical features from each column of a DataFrame.
    Columns are different signals (e.g. flights), and rows are samples.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each column is a signal (e.g. one flight).

    Returns
    -------
    pd.DataFrame
        Features of shape (n_columns, 10).
        Columns: ['rms','mean','std','peak2peak','kurtosis','skewness',
                  'crest','clearance','shape','impulse']
    """
    num_cols = df.shape[1]
    num_features = 10
    features = np.zeros((num_cols, num_features))

    for i, col in enumerate(df.columns):
        data = df[col].dropna().to_numpy(dtype=float)
        if len(data) == 0:
            continue  # skip empty column

        rms = np.sqrt(np.mean(data ** 2))
        features[i, 0] = rms                                  # rms
        features[i, 1] = np.mean(data)                         # mean
        features[i, 2] = np.std(data)                          # std
        features[i, 3] = data.max() - data.min()               # peak2peak
        features[i, 4] = kurtosis(data, fisher=False)          # kurtosis
        features[i, 5] = skew(data)                            # skewness
        features[i, 6] = abs(data.max()) / rms                 # crest
        features[i, 7] = abs(data.max()) / (np.mean(np.sqrt(np.abs(data))) ** 2)  # clearance
        features[i, 8] = rms / np.mean(np.abs(data))           # shape
        features[i, 9] = abs(data.max()) / np.mean(np.abs(data))  # impulse

    feature_names = [
        "rms","mean","std","peak2peak","kurtosis","skewness",
        "crest","clearance","shape","impulse"
    ]
    return pd.DataFrame(features, index=df.columns, columns=feature_names)


### Define features name
feature_name=['1-TO-Y-rms','2-TO-Y-mean','3-TO-Y-std','4-TO-Y-peak2peak','5-TO-Y-kurtosis','6-TO-Y-skewness','7-TO-Y-crest_indicator','8-TO-Y-clearance_indicator','9-TO-Y-shape_indicator','10-TO-Y-impulse_indicator',
              '11-TO-Z-rms','12-TO-Z-mean','13-TO-Z-std','14-TO-Z-peak2peak','15-TO-Z-kurtosis','16-TO-Z-skewness','17-TO-Z-crest_indicator','18-TO-Z-clearance_indicator','19-TO-Z-shape_indicator','20-TO-Z-impulse_indicator',
              '21-LD-Y-rms','22-LD-Y-mean','23-LD-Y-std','24-LD-Y-peak2peak','25-LD-Y-kurtosis','26-LD-Y-skewness','27-LD-Y-crest_indicator','28-LD-Y-clearance_indicator','29-LD-Y-shape_indicator','30-LD-Y-impulse_indicator',
              '31-LD-Z-rms','32-LD-Z-mean','33-LD-Z-std','34-LD-Z-peak2peak','35-LD-Z-kurtosis','36-LD-Z-skewness','37-LD-Z-crest_indicator','38-LD-Z-clearance_indicator','39-LD-Z-shape_indicator','40-LD-Z-impulse_indicator']
