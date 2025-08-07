# combined_api.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# === Cài đặt chung ===
checkpoint_dir = 'model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === GRU Model ===
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.linear(out[:, -1, :])

# === Load GRU model và scaler ===
gru_model = GRUModel(input_size=9).to(device)
gru_model.load_state_dict(torch.load(f"{checkpoint_dir}/best_model.pth", map_location=device))
gru_model.eval()
with open(f"{checkpoint_dir}/input_scaler.pkl", "rb") as f:
    gru_input_scaler = pickle.load(f)
with open(f"{checkpoint_dir}/target_scaler.pkl", "rb") as f:
    gru_target_scaler = pickle.load(f)

# === Load XGBoost model và scaler ===
with open("model_checkpoints/best_ml_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("model_checkpoints/ml_scaler.pkl", "rb") as f:
    xgb_scaler = pickle.load(f)

# === FastAPI app ===
app = FastAPI(title="Combined GRU & XGBoost API")

# === GRU Input ===
class GRUInput(BaseModel):
    data: List[List[float]]  # 10 tuần, 8 đặc trưng

# === XGB Input ===
class XGBInput(BaseModel):
    features: List[float]  # 10 tuần * 8 đặc trưng = 80

# === GRU Prediction Endpoint ===
@app.post("/predict/gru")
def predict_gru(input_data: GRUInput):
    try:
        if len(input_data.data) != 10:
            raise HTTPException(status_code=400, detail="❌ Cần đúng 10 tuần dữ liệu (10 dòng).")

        input_df = pd.DataFrame(input_data.data, columns=[
            'Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
            'CPI', 'Unemployment', 'WeekOfYear', 'Month'
        ])

        input_df['slope_3w'] = input_df['Weekly_Sales'].diff().rolling(window=3).mean().fillna(0)

        scaled_inputs_only = gru_input_scaler.transform(input_df.iloc[:, 1:])
        scaled_weekly_sales = gru_target_scaler.transform(np.log1p(input_df[['Weekly_Sales']]))

        scaled_input = np.concatenate([scaled_weekly_sales, scaled_inputs_only], axis=1)
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred_scaled = gru_model(input_tensor).detach().cpu().numpy()

        y_pred_real = np.expm1(gru_target_scaler.inverse_transform(y_pred_scaled)[0, 0])
        return {"predicted_weekly_sales": round(float(y_pred_real), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === XGBoost Prediction Endpoint ===
@app.post("/predict/xgb")
def predict_xgb(request: XGBInput):
    try:
        if len(request.features) != 80:
            return {"error": f"Số lượng đặc trưng không hợp lệ. Cần 80 giá trị."}

        input_array = np.array(request.features).reshape(10, 8)
        flattened = input_array.flatten().reshape(1, -1)
        input_scaled = xgb_scaler.transform(flattened)

        prediction_log = xgb_model.predict(input_scaled)[0]
        prediction_real = np.expm1(prediction_log)

        return {
            "predicted_weekly_sales": round(float(prediction_real), 2),
            "debug_info": {
                "log_prediction": float(prediction_log),
                "real_prediction": float(prediction_real)
            }
        }
    except Exception as e:
        return {"error": f"Lỗi trong quá trình dự đoán: {str(e)}"}

# === Optional: test XGB model with real data ===
@app.get("/test/xgb")
def test_prediction():
    df = pd.read_csv("walmart_processed_by_week.csv")
    store_df = df[df['Store'] == 1].sort_values('Week_Index')
    feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
    data = store_df[feature_cols].values.astype(np.float32)[-10:]

    flattened = data.flatten().reshape(1, -1)
    input_scaled = xgb_scaler.transform(flattened)

    prediction_log = xgb_model.predict(input_scaled)[0]
    prediction_real = np.expm1(prediction_log)

    return {
        "predicted_weekly_sales": round(float(prediction_real), 2),
        "log_prediction": float(prediction_log)
    }

# === Run app ===
if __name__ == "__main__":
    uvicorn.run("allapi:app", host="0.0.0.0", port=10002, reload=True)
