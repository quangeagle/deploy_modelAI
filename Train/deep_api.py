# deep_api.py - FastAPI cho GRU model
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import uvicorn
from sklearn.preprocessing import MinMaxScaler

# ========== 1. Model Definition ==========
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.linear(out[:, -1, :])

# ========== 2. Load Model & Scalers ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# Load model
model = GRUModel(input_size=8)  # 8 features
model.load_state_dict(torch.load("model_checkpoints/best_model.pth", map_location=device))
model.to(device)
model.eval()

# Load scalers
with open("model_checkpoints/input_scaler.pkl", "rb") as f:
    input_scaler = pickle.load(f)
with open("model_checkpoints/target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

print("✅ Đã load GRU model và scalers thành công!")

# ========== 3. FastAPI App ==========
app = FastAPI(title="GRU Walmart Sales Prediction API")

# Request schemas
class WeeklyInput(BaseModel):
    data: List[List[float]]  # 10 tuần x 8 features

class TestInput(BaseModel):
    store_id: int = 1  # Store để test

@app.get("/")
def root():
    return {
        "message": "GRU Walmart Sales Prediction API",
        "model": "GRU (Gated Recurrent Unit)",
        "input_format": "10 weeks x 8 features",
        "features": ["Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment", "WeekOfYear", "Month"]
    }

@app.post("/predict")
def predict_sales(input_data: WeeklyInput):
    """Dự đoán doanh số tuần tiếp theo"""
    try:
        if len(input_data.data) != 10:
            raise HTTPException(status_code=400, detail="❌ Cần đúng 10 tuần dữ liệu (10 dòng).")
        
        feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
                        'CPI', 'Unemployment', 'WeekOfYear', 'Month']
        
        # Chuyển thành DataFrame
        input_df = pd.DataFrame(input_data.data, columns=feature_cols)
        
        # Tách Weekly_Sales và các features khác
        input_features = feature_cols[1:]  # Bỏ Weekly_Sales
        weekly_sales_col = ['Weekly_Sales']

        # Scale dữ liệu
        scaled_inputs_only = input_scaler.transform(input_df[input_features])
        scaled_weekly_sales = target_scaler.transform(input_df[weekly_sales_col])
        
        # Ghép lại: Weekly_Sales trước + các features sau
        scaled_input = np.concatenate([scaled_weekly_sales, scaled_inputs_only], axis=1)

        # Dự đoán
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            y_pred_scaled = model(input_tensor).detach().cpu().numpy()
        
        # Inverse transform để lấy giá trị thật
        y_pred_real = target_scaler.inverse_transform(y_pred_scaled)[0, 0]

        return {
            "predicted_weekly_sales": round(float(y_pred_real), 2),
            "debug_info": {
                "scaled_prediction": float(y_pred_scaled[0, 0]),
                "input_shape": scaled_input.shape,
                "model_type": "GRU"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi dự đoán: {str(e)}")

@app.get("/test")
def test_prediction():
    """Test với dữ liệu thực tế từ store 1"""
    try:
        # Load dữ liệu
        df = pd.read_csv("walmart_processed_by_week.csv")
        
        # Lấy 10 tuần cuối của store 1
        store_df = df[df['Store'] == 1].sort_values('Week_Index')
        feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
        data = store_df[feature_cols].values.astype(np.float32)
        
        # Lấy 10 tuần cuối
        last_10_weeks = data[-10:]
        actual_target = data[-1, 0]  # Tuần cuối
        
        # Tách Weekly_Sales và features
        input_features = feature_cols[1:]
        weekly_sales_col = ['Weekly_Sales']
        
        # Scale
        scaled_inputs_only = input_scaler.transform(last_10_weeks[:, 1:])
        scaled_weekly_sales = target_scaler.transform(last_10_weeks[:, 0:1])
        scaled_input = np.concatenate([scaled_weekly_sales, scaled_inputs_only], axis=1)
        
        # Predict
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            y_pred_scaled = model(input_tensor).detach().cpu().numpy()
        y_pred_real = target_scaler.inverse_transform(y_pred_scaled)[0, 0]
        
        return {
            "test_data": {
                "actual_weekly_sales": float(actual_target),
                "predicted_weekly_sales": round(float(y_pred_real), 2),
                "error": round(float(abs(y_pred_real - actual_target)), 2),
                "error_percentage": round(float(abs(y_pred_real - actual_target)/actual_target*100), 2)
            },
            "debug": {
                "scaled_prediction": float(y_pred_scaled[0, 0]),
                "input_range": {
                    "min": float(last_10_weeks.min()),
                    "max": float(last_10_weeks.max())
                },
                "model_type": "GRU"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi test: {str(e)}")

@app.get("/model_info")
def get_model_info():
    """Thông tin về model"""
    return {
        "model_type": "GRU (Gated Recurrent Unit)",
        "architecture": {
            "input_size": 8,
            "hidden_size": 64,
            "num_layers": 1
        },
        "device": str(device),
        "scalers": {
            "input_scaler": "MinMaxScaler",
            "target_scaler": "MinMaxScaler"
        },
        "input_format": "10 weeks x 8 features",
        "output": "Weekly Sales prediction (USD)"
    }

# ========== 4. Run Server ==========
if __name__ == "__main__":
    print("🚀 Starting GRU API server...")
    print("📊 Model loaded successfully!")
    print("🌐 API will be available at: http://localhost:8002")
    print("📝 Test endpoints:")
    print("   GET  / - API info")
    print("   GET  /test - Test with real data")
    print("   POST /predict - Make prediction")
    print("   GET  /model_info - Model details")
    
    uvicorn.run("deep_api:app", host="0.0.0.0", port=8002, reload=True) 