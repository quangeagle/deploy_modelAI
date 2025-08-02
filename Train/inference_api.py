from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import torch
import pickle
from deep_walmart import GRUModel  # Nếu bạn đã tách GRU ra, còn không thì paste lại lớp GRUNet ở đây
import os
# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_dir = 'model_checkpoints'
print("✅ Danh sách file:", os.listdir(checkpoint_dir))
# --- Load model ---
model = GRUModel(input_size=8, hidden_size=64, num_layers=1).to(device)

model.load_state_dict(torch.load(f"{checkpoint_dir}/best_model.pth", map_location=device))
model.eval()

# --- Load scalers ---
with open(f"{checkpoint_dir}/input_scaler.pkl", 'rb') as f:
    input_scaler = pickle.load(f)

with open(f"{checkpoint_dir}/target_scaler.pkl", 'rb') as f:
    target_scaler = pickle.load(f)

# --- FastAPI app ---
app = FastAPI()

class WeeklyInput(BaseModel):
    data: List[List[float]]  # 10 tuần x 8 giá trị

@app.post("/predict")
def predict_sales(input_data: WeeklyInput):
    try:
        if len(input_data.data) != 10:
            raise HTTPException(status_code=400, detail="Cần đúng 10 tuần dữ liệu.")

        feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
                        'CPI', 'Unemployment', 'WeekOfYear', 'Month']
        df = pd.DataFrame(input_data.data, columns=feature_cols)

        # Tách cột
        X = df.drop(columns=['Weekly_Sales'])
        y = df[['Weekly_Sales']]

        # Scale
        X_scaled = input_scaler.transform(X)
        y_scaled = target_scaler.transform(y)
        final_input = np.concatenate([y_scaled, X_scaled], axis=1)

        # Convert to tensor
        input_tensor = torch.tensor(final_input, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().numpy()
            pred = target_scaler.inverse_transform(pred_scaled)[0, 0]

        return {"predicted_weekly_sales": pred}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Run if main ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_api:app", host="0.0.0.0", port=8000, reload=True)
