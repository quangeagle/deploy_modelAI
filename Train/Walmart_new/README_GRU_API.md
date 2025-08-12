# 🧠 GRU Sales Prediction API

API để dự đoán doanh thu tuần tiếp theo dựa trên doanh thu 10 tuần trước sử dụng mô hình GRU.

## 📊 Model Performance

- **R² Score**: 0.9650 (96.5% accuracy)
- **RMSE**: $57,409
- **MAE**: $36,647
- **Features**: Chỉ sử dụng Weekly_Sales (doanh thu tuần)

## 🚀 Quick Start

### 1. Khởi động API

```bash
cd Train/Walmart_new
python gru_api.py
```

API sẽ chạy tại: `http://localhost:8000`

### 2. Test API

```bash
python test_gru_api.py
```

### 3. Truy cập Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📋 API Endpoints

### 1. Health Check
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

### 2. Model Info
```bash
GET /model-info
```

**Response:**
```json
{
  "model_type": "GRU",
  "input_size": 1,
  "hidden_size": 128,
  "num_layers": 2,
  "dropout": 0.2,
  "lookback_period": 10,
  "features": ["Weekly_Sales"],
  "training_metrics": {
    "r2_score": 0.9650,
    "rmse": 57409.36,
    "mae": 36646.54
  }
}
```

### 3. Example
```bash
GET /example
```

**Response:**
```json
{
  "example_request": {
    "sales_history": [1000000, 1050000, 1100000, 1150000, 1200000, 1250000, 1300000, 1350000, 1400000, 1450000],
    "store_id": 1
  },
  "expected_response": {
    "predicted_sales": 1500000,
    "confidence_score": 0.85,
    "input_sequence": [1000000, 1050000, 1100000, 1150000, 1200000, 1250000, 1300000, 1350000, 1400000, 1450000],
    "message": "Dự đoán thành công"
  }
}
```

### 4. Prediction (Main Endpoint)
```bash
POST /predict
```

**Request Body:**
```json
{
  "sales_history": [1000000, 1050000, 1100000, 1150000, 1200000, 1250000, 1300000, 1350000, 1400000, 1450000]
}
```

**Response:**
```json
{
  "predicted_sales": 1500000.0,
  "confidence_score": 0.85,
  "input_sequence": [1000000, 1050000, 1100000, 1150000, 1200000, 1250000, 1300000, 1350000, 1400000, 1450000],
  "message": "Dự đoán thành công"
}
```

## 🧪 Test Scenarios

### 1. Trending Up (Xu hướng tăng)
```json
{
  "sales_history": [1000000, 1050000, 1100000, 1150000, 1200000, 1250000, 1300000, 1350000, 1400000, 1450000]
}
```

### 2. Trending Down (Xu hướng giảm)
```json
{
  "sales_history": [1500000, 1450000, 1400000, 1350000, 1300000, 1250000, 1200000, 1150000, 1100000, 1050000]
}
```

### 3. Stable (Ổn định)
```json
{
  "sales_history": [1200000, 1200000, 1200000, 1200000, 1200000, 1200000, 1200000, 1200000, 1200000, 1200000]
}
```

### 4. Volatile (Biến động)
```json
{
  "sales_history": [1000000, 1500000, 800000, 1600000, 900000, 1400000, 1100000, 1300000, 1200000, 1400000]
}
```

## 🔧 Technical Details

### Model Architecture
- **Type**: GRU (Gated Recurrent Unit)
- **Input Size**: 1 (Weekly_Sales only)
- **Hidden Size**: 128
- **Num Layers**: 2
- **Dropout**: 0.2
- **Lookback Period**: 10 weeks

### Data Processing
1. **Input**: 10 weeks of sales history
2. **Scaling**: MinMaxScaler for both input and output
3. **Prediction**: Single value for next week's sales
4. **Confidence**: Based on input stability (coefficient of variation)

### Files Required
- `best_gru_model.pth` - Trained GRU model
- `sequence_scaler.pkl` - Input scaler
- `target_scaler.pkl` - Output scaler

## 🎯 Usage Examples

### Python Requests
```python
import requests

# Test prediction
payload = {
    "sales_history": [1000000, 1050000, 1100000, 1150000, 1200000, 
                      1250000, 1300000, 1350000, 1400000, 1450000]
}

response = requests.post("http://localhost:8000/predict", json=payload)
result = response.json()

print(f"Predicted Sales: ${result['predicted_sales']:,.2f}")
print(f"Confidence: {result['confidence_score']:.3f}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sales_history": [1000000, 1050000, 1100000, 1150000, 1200000, 1250000, 1300000, 1350000, 1400000, 1450000]
     }'
```

## 📈 Business Logic

### How GRU Works
1. **Sequential Learning**: Học patterns từ 10 tuần trước
2. **Trend Analysis**: Nhận diện xu hướng tăng/giảm
3. **Seasonal Patterns**: Học chu kỳ mùa vụ
4. **Autocorrelation**: Doanh thu tuần này ảnh hưởng tuần sau

### Confidence Score
- **High Confidence (>0.8)**: Input ổn định, trend rõ ràng
- **Medium Confidence (0.6-0.8)**: Input có biến động vừa phải
- **Low Confidence (<0.6)**: Input biến động mạnh, khó dự đoán

## 🚨 Error Handling

### Common Errors
1. **Invalid Input Length**: Cần đúng 10 giá trị
2. **Negative Values**: Doanh thu không được âm
3. **Model Not Loaded**: Kiểm tra file model tồn tại
4. **Server Error**: Kiểm tra logs

### Error Response
```json
{
  "detail": "Lỗi dự đoán: Cần đúng 10 giá trị doanh thu tuần trước"
}
```

## 🔍 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Model Info
```bash
curl http://localhost:8000/model-info
```

## 📝 Notes

- **Performance**: R² = 96.5% cho thấy model rất hiệu quả
- **Simplicity**: Chỉ cần doanh thu quá khứ, không cần features khác
- **Real-time**: API có thể xử lý real-time predictions
- **Scalable**: Có thể deploy trên production với load balancing

## 🎯 Next Steps

1. **Production Deployment**: Deploy lên cloud (AWS, GCP, Azure)
2. **Load Balancing**: Sử dụng nginx hoặc load balancer
3. **Monitoring**: Thêm logging và monitoring
4. **Caching**: Implement Redis cache cho performance
5. **Authentication**: Thêm API key authentication
6. **Rate Limiting**: Giới hạn request rate
