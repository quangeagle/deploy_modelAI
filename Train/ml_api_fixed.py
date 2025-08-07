from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import uvicorn

# Load model và input scaler
with open("model_checkpoints/best_ml_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model_checkpoints/ml_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI(title="XGBoost Weekly Sales Prediction API")

FEATURE_COUNT = 80  # 10 tuần * 8 đặc trưng

class PredictRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict_sales(request: PredictRequest):
    if len(request.features) != FEATURE_COUNT:
        return {"error": f"Số lượng đặc trưng không hợp lệ. Cần {FEATURE_COUNT} giá trị."}

    try:
        # ✅ BỔ SUNG: reshape input thành (10 tuần, 8 đặc trưng), rồi flatten lại
        input_array = np.array(request.features).reshape(10, 8)
        
        # ❌ KHÔNG log1p vì model đã được train trên dữ liệu gốc
        flattened = input_array.flatten().reshape(1, -1)
        
        # Scale
        input_scaled = scaler.transform(flattened)

        # Dự đoán
        prediction_log = model.predict(input_scaled)[0]

        # Trả về giá trị thực
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

# Optional test route
@app.get("/test")
def test_prediction():
    import pandas as pd
    df = pd.read_csv("walmart_processed_by_week.csv")
    store_df = df[df['Store'] == 1].sort_values('Week_Index')
    feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
    data = store_df[feature_cols].values.astype(np.float32)[-10:]

    # ❌ KHÔNG log1p vì model train trên dữ liệu gốc
    flattened = data.flatten().reshape(1, -1)
    input_scaled = scaler.transform(flattened)

    prediction_log = model.predict(input_scaled)[0]
    prediction_real = np.expm1(prediction_log)

    return {
        "predicted_weekly_sales": round(float(prediction_real), 2),
        "log_prediction": float(prediction_log)
    }

if __name__ == "__main__":
    uvicorn.run("ml_api_fixed:app", host="0.0.0.0", port=8001, reload=True)
