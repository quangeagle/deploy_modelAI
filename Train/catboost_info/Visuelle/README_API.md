# 🚀 Walmart Revenue Prediction API

FastAPI service để dự đoán doanh thu tuần tiếp theo cho Walmart.

## 📋 Cài đặt

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Chạy API

```bash
python walmart_revenue_api.py
```

API sẽ chạy tại: `http://localhost:8000`

## 📚 API Endpoints

### 1. **Health Check**
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### 2. **Predict từ JSON**
```bash
POST /predict
```

**Request Body:**
```json
{
  "weekly_sales": [1000.0, 1200.0, 1100.0, 1300.0, 1400.0, 1350.0, 1500.0, 1600.0, 1550.0, 1700.0],
  "holiday_flag": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  "temperature": [42.31, 38.51, 39.93, 46.63, 46.50, 57.79, 58.85, 57.92, 54.58, 51.45],
  "fuel_price": [2.572, 2.548, 2.514, 2.561, 2.625, 2.667, 2.720, 2.732, 2.710, 2.699],
  "cpi": [211.0963582, 211.2421697, 211.2891409, 211.319415, 211.350625, 211.4281144, 211.6522349, 212.1662683, 212.7728543, 213.2801017],
  "unemployment": [8.106, 8.106, 8.106, 8.106, 8.106, 8.106, 8.106, 8.106, 8.106, 8.106],
  "week_of_year": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  "month": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```

**Response:**
```json
{
  "predicted_weekly_sales": 1750.25,
  "confidence_score": 0.85,
  "model_info": {
    "model_type": "GRU",
    "input_features": 8,
    "device": "cpu",
    "scaler_type": "MinMaxScaler"
  }
}
```

### 3. **Predict từ Excel File**
```bash
POST /predict-file
```

**Request:** Upload Excel file với 10 tuần dữ liệu

**File format:**
| Weekly_Sales | Holiday_Flag | Temperature | Fuel_Price | CPI | Unemployment | WeekOfYear | Month |
|-------------|-------------|-------------|------------|-----|-------------|------------|-------|
| 1000.0 | 0 | 42.31 | 2.572 | 211.096 | 8.106 | 1 | 1 |
| 1200.0 | 0 | 38.51 | 2.548 | 211.242 | 8.106 | 2 | 1 |
| ... | ... | ... | ... | ... | ... | ... | ... |

### 4. **Reload Model**
```bash
POST /reload-model
```

**Response:**
```json
{
  "message": "Model reloaded successfully",
  "model_loaded": true
}
```

## 🔧 Cấu trúc thư mục

```
Train/catboost_info/Visuelle/
├── walmart_revenue_api.py      # FastAPI service
├── requirements.txt            # Dependencies
├── model_checkpoints/         # Model files
│   ├── best_model.pth
│   ├── input_scaler.pkl
│   └── target_scaler.pkl
└── README_API.md             # Hướng dẫn này
```

## 🌐 Swagger UI

Truy cập: `http://localhost:8000/docs`

## 📝 Ví dụ sử dụng với Python

```python
import requests
import json

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict
data = {
    "weekly_sales": [1000.0, 1200.0, 1100.0, 1300.0, 1400.0, 1350.0, 1500.0, 1600.0, 1550.0, 1700.0],
    "holiday_flag": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "temperature": [42.31, 38.51, 39.93, 46.63, 46.50, 57.79, 58.85, 57.92, 54.58, 51.45],
    "fuel_price": [2.572, 2.548, 2.514, 2.561, 2.625, 2.667, 2.720, 2.732, 2.710, 2.699],
    "cpi": [211.0963582, 211.2421697, 211.2891409, 211.319415, 211.350625, 211.4281144, 211.6522349, 212.1662683, 212.7728543, 213.2801017],
    "unemployment": [8.106, 8.106, 8.106, 8.106, 8.106, 8.106, 8.106, 8.106, 8.106, 8.106],
    "week_of_year": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "month": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()
print(f"Predicted revenue: ${result['predicted_weekly_sales']:.2f}")
```

## 🚀 Deploy

### Local
```bash
python walmart_revenue_api.py
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn walmart_revenue_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "walmart_revenue_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## ⚠️ Lưu ý

1. **Model files**: Đảm bảo có file model trong `model_checkpoints/`
2. **Input validation**: API yêu cầu đúng 10 tuần dữ liệu
3. **File format**: Excel file phải có đúng 8 cột
4. **Memory**: Model sẽ load vào memory khi startup

## 🔍 Troubleshooting

- **Model not loaded**: Kiểm tra file model trong `model_checkpoints/`
- **Validation error**: Đảm bảo input có đúng format
- **File error**: Kiểm tra Excel file có đúng cột và 10 rows