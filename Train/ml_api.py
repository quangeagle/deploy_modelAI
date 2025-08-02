# ml_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import uvicorn

# Load mô hình và scaler
with open("model_checkpoints/best_ml_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model_checkpoints/ml_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define FastAPI app
app = FastAPI(title="XGBoost Weekly Sales Prediction API")

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

    # Dự đoán
    prediction = model.predict(input_scaled)[0]
    return {
        "predicted_weekly_sales": round(float(prediction), 2)
    }

# Run server
if __name__ == "__main__":
    uvicorn.run("ml_api:app", host="0.0.0.0", port=8000, reload=True)
