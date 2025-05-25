# import numpy as np
# from timesfm.timesfm_torch import TimesFmTorch

# def main():
#     model = TimesFmTorch(
#         context_len=512,
#         horizon_len=128,
#         input_patch_len=32,
#         output_patch_len=128,
#         num_layers=20,
#         model_dims=1280,
#         backend="pytorch"
#     )

#     model.load_from_checkpoint(repo_id="google/timesfm-1.0-200m-pytorch")

#     forecast_input = [np.sin(np.linspace(0, 20, 512))]
#     frequency_input = [0]  # daily

#     point_forecast, _ = model.forecast(forecast_input, freq=frequency_input)

#     print("預測結果（前10筆）:")
#     print(point_forecast[0][:10])

# if __name__ == "__main__":
#     main()







# import numpy as np
# from timesfm.timesfm_torch import TimesFmTorch

# def main():
#     model = TimesFmTorch()

#     # 手動指定設定參數
#     model.context_len = 512
#     model.horizon_len = 128
#     model.input_patch_len = 32
#     model.output_patch_len = 128
#     model.num_layers = 20
#     model.model_dims = 1280
#     model.num_heads = 16  # 需要與 model_dims 對應可被整除
#     model.quantiles = []
#     model.per_core_batch_size = 1
#     model.use_pos_emb = True
#     model.backend = "cpu"  # 或 "gpu" 若有 CUDA

#     model.__post_init__()  # 必須手動呼叫，才會建立模型結構

#     model.load_from_checkpoint(repo_id="google/timesfm-1.0-200m-pytorch")

#     forecast_input = [np.sin(np.linspace(0, 20, 512))]
#     frequency_input = [0]

#     point_forecast, _ = model.forecast(
#         forecast_input,
#         freq=frequency_input,
#         prediction_length=128
#     )

#     print("預測前 10 筆結果：")
#     print(point_forecast[0][:10])

# if __name__ == "__main__":
#     main()











import numpy as np
from timesfm import timesfm_base
from timesfm.timesfm_torch import TimesFmTorch

def main():
    # 設定 hparams（你可以自行調整參數）
    hparams = timesfm_base.TimesFmHparams(
        context_len=512,
        horizon_len=128,
        input_patch_len=32,
        output_patch_len=1280,
        num_layers=20,
        model_dims=1280,
        num_heads=16,
        quantiles=[0.5],
        # use_pos_emb=True,
        use_positional_embedding=True,
        per_core_batch_size=1,
        backend="cpu"  # 或 "gpu" 若你有 CUDA
    )

    # 設定 checkpoint 來源（從 Hugging Face 自動下載）
    checkpoint = timesfm_base.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    )

    # 建立模型
    model = TimesFmTorch(hparams=hparams, checkpoint=checkpoint)

    # 載入模型權重
    model.load_from_checkpoint(checkpoint)

    # 測試資料
    forecast_input = [np.sin(np.linspace(0, 20, 512))]
    frequency_input = [0]

    # 預測
    point_forecast, _ = model.forecast(
        forecast_input,
        freq=frequency_input,
    )

    print("預測結果（前 10 筆）：")
    print(point_forecast[0][:10])

if __name__ == "__main__":
    main()









# from huggingface_hub import snapshot_download
# import torch

# path = snapshot_download("google/timesfm-1.0-200m-pytorch")
# checkpoint = torch.load(f"{path}/torch_model.ckpt", map_location="cpu")
# print(checkpoint.keys())