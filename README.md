# YOLOv8-Object-Detection

Go through a machine learning application including task identification, data collection, model training, model evaluation, and deployment of a trained model.

## 建立並啟動虛擬環境

```bash
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# 安裝依賴
pip install -r requirements.txt
```

## 使用方式

```bash
# 切換資料集（例如 dataset_fruit262）:
python train.py --dataset dataset_fruit-detector

# 復原訓練模型流程
yolo train resume model=runs/fruit_detector_v2/weights/last.pt
```

## 資料集準備

如果原始資料是「每個類別一個資料夾」的結構，可先轉成 YOLO detection 格式，並產生 `train/valid/test` 與 `data.yaml`。

```bash
# 使用預設來源與輸出路徑
python tools/prepare_detection_dataset.py

# 明確指定來源與輸出
python tools/prepare_detection_dataset.py --source-dir dataset-Fruits-262/Fruit-262 --output-dir dataset_fruit262
```

預設會自動嘗試找到常見來源資料夾，例如 `dataset-Fruits-262/Fruit-262`。

## 啟動網頁介面:

```bash
python app.py
```

## 專案目錄

```text
web/
	app.py
	yolo_utils.py
	templates/
		index.html

train.py        # 訓練入口
predict.py      # 推論入口
app.py          # 相容入口（轉發到 web/app.py）
yolo_utils.py   # 相容匯出（轉發到 web/yolo_utils.py）
```