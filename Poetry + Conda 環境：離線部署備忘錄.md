# Poetry + Conda 環境：離線部署備忘錄

## ✨ 概述
本文件說明如何在 Conda 環境中使用 Poetry 管理 Python 套件，並將整個環境部署至無網路的 offline 系統。

---

## ⚡ Step 1: 建立 Conda 環境並使用 Poetry 安裝

```bash
# 建立 Conda 環境
conda create -n myproj-env python=3.10 -y
conda activate myproj-env

# 設定 Poetry 使用目前 Conda 環境，不建立 virtualenv
poetry config virtualenvs.create false

# 切換至專案資料夾（含 pyproject.toml）
cd D:\your_project_folder

# 安裝 Poetry 相依套件
poetry install
```

---

## 🔍 Step 2: 驗證環境是否正常

```bash
# 確認 Python 套件是否能正常匯入
python -c "import torch; print(torch.__version__)"
```

---

## 📦 Step 3: 將環境打包搬移

### ✅ 方法一：直接複製 Conda 環境資料夾
將以下資料夾整包複製到 offline 機器上對應位置：
```
D:\Anaconda3\envs\myproj-env\
```

### ✅ 方法二：使用 conda-pack

```bash
conda activate myproj-env
conda install -c conda-forge conda-pack
conda-pack -n myproj-env -o myproj-env.tar.gz
```

**在 offline 機器中解壓縮：**
```bash
mkdir -p ~/myproj-env
# 或使用 Windows 的壓縮軟體解壓縮

tar -xzf myproj-env.tar.gz -C ~/myproj-env
```

**啟動方式：**
```bash
# UNIX/Linux/macOS
source ~/myproj-env/bin/activate

# Windows
cd myproj-env\Scripts
activate.bat
```

---

## 💾 Step 4: Offline 環境使用說明

```bash
# 啟動 Conda 環境
conda activate myproj-env

# 執行你的模型程式
python run_model.py
```

---

## ✨ 附加建議：匯出 requirements.txt（選用）

若你希望改用 pip 安裝，也可用 poetry 匯出需求檔：

```bash
poetry export -f requirements.txt --without-hashes > requirements.txt
```

然後配合以下指令，在 offline 環境安裝：
```bash
pip install --no-index --find-links=pkgs -r requirements.txt
```

---

## 📍 備註
- 適用於 offline 實驗平台開發預訓練模型（如 Google TimesFM, Huggingface 模型等）
- 推薦使用 `conda-pack` 封裝更保險，特別當有 native/CUDA 相依套件時

