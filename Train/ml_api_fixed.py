# ml_api_fixed.py - API đã sửa với target scaling
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import uvicorn

# Load mô hình và scalers
with open("model_checkpoints/best_ml_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model_checkpoints/ml_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model_checkpoints/target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

# Define FastAPI app
app = FastAPI(title="XGBoost Weekly Sales Prediction API (Fixed)")

# Số lượng đặc trưng đầu vào (lookback = 10, 8 đặc trưng → 80 features)
FEATURE_COUNT = 80

# Request schema
class PredictRequest(BaseModel):
    features: list[float]  # 80 giá trị đã sắp theo thứ tự thời gian (10 tuần gần nhất)

@app.post("/predict")
def predict_sales(request: PredictRequest):
    if len(request.features) != FEATURE_COUNT:
        return {"error": f"Số lượng đặc trưng không hợp lệ. Cần {FEATURE_COUNT} giá trị."}
    
    # Chuyển thành mảng numpy và chuẩn hóa
    input_array = np.array(request.features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # Dự đoán (kết quả đã scale)
    prediction_scaled = model.predict(input_scaled)[0]
    
    # Inverse transform để lấy giá trị thực
    prediction_real = target_scaler.inverse_transform([[prediction_scaled]])[0][0]
    
    return {
        "predicted_weekly_sales": round(float(prediction_real), 2),
        "debug_info": {
            "scaled_prediction": float(prediction_scaled),
            "real_prediction": float(prediction_real)
        }
    }

# Test endpoint
@app.get("/test")
def test_prediction():
    """Test với dữ liệu mẫu từ store 1"""
    import pandas as pd
    
    # Load dữ liệu
    df = pd.read_csv("walmart_processed_by_week.csv")
    
    # Lấy 10 tuần cuối của store 1
    store_df = df[df['Store'] == 1].sort_values('Week_Index')
    feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
    data = store_df[feature_cols].values.astype(np.float32)
    
    # Lấy 10 tuần cuối
    last_10_weeks = data[-10:]
    test_sequence = last_10_weeks.flatten()
    actual_target = data[-1, 0]  # Tuần cuối
    
    # Predict
    input_scaled = scaler.transform(test_sequence.reshape(1, -1))
    prediction_scaled = model.predict(input_scaled)[0]
    prediction_real = target_scaler.inverse_transform([[prediction_scaled]])[0][0]
    
    return {
        "test_data": {
            "actual_weekly_sales": float(actual_target),
            "predicted_weekly_sales": round(float(prediction_real), 2),
            "error": round(float(abs(prediction_real - actual_target)), 2),
            "error_percentage": round(float(abs(prediction_real - actual_target)/actual_target*100), 2)
        },
        "debug": {
            "scaled_prediction": float(prediction_scaled),
            "input_range": {
                "min": float(test_sequence.min()),
                "max": float(test_sequence.max())
            }
        }
    }

# Run server
if __name__ == "__main__":
    uvicorn.run("ml_api_fixed:app", host="0.0.0.0", port=8001, reload=True) 