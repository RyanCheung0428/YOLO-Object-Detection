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
# 切換資料集（例如 dataset2）:
python train.py --dataset dataset2

# 圖片推論:
python predict.py --image test_image/your_image.jpg
```

啟動網頁介面:

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