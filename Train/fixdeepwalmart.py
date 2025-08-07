

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
import matplotlib.pyplot as plt

# === Cài đặt ===
checkpoint_dir = 'model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
r2_path = os.path.join(checkpoint_dir, "best_r2_score.txt")
best_r2_score = float('-inf')
if os.path.exists(r2_path):
    with open(r2_path, 'r') as f:
        best_r2_score = float(f.read().strip())

# === Dataset ===
class WalmartDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

# === GRU Model ===
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.linear(out[:, -1, :])

# === Sequence generator ===
def create_sequences(data, lookback):
    sequences, targets = [], []
    for i in range(len(data) - lookback):
        seq = data[i:i+lookback]
        target = data[i+lookback, 0]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# === Synthetic Data Generation ===
def generate_trend_samples():
    synthetic = []
    for start in [100000, 200000, 300000]:
        # Increasing
        inc = [[v, 0, 30, 3.0, 220, 6.0, w, 6, (v-start)/10000] for w, v in enumerate(range(start, start + 100000, 10000))]
        # Decreasing
        dec = [[v, 0, 30, 3.0, 220, 6.0, w, 6, -(v-start)/10000] for w, v in enumerate(range(start + 100000, start, -10000))]
        synthetic.extend(inc + dec)
    return pd.DataFrame(synthetic, columns=['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price','CPI', 'Unemployment', 'WeekOfYear', 'Month', 'slope_3w'])

# === Train ===
def train_model(df, lookback=12):
    df['slope_3w'] = df['Weekly_Sales'].diff().rolling(window=3).mean().fillna(0)
    synthetic_df = generate_trend_samples()
    df = pd.concat([df, synthetic_df], ignore_index=True)

    feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month', 'slope_3w']
    df['Weekly_Sales'] = np.log1p(df['Weekly_Sales'])

    input_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    input_features = feature_cols[1:]
    df[input_features] = input_scaler.fit_transform(df[input_features])
    df[['Weekly_Sales']] = target_scaler.fit_transform(df[['Weekly_Sales']])

    all_sequences, all_targets = [], []
    for store_id in df['Store'].unique():
        store_df = df[df['Store'] == store_id].sort_values('Week_Index') if 'Store' in df else df.sort_values('WeekOfYear')
        data = store_df[feature_cols].values.astype(np.float32)
        X, y = create_sequences(data, lookback)
        all_sequences.extend(X)
        all_targets.extend(y)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(np.array(all_sequences), np.array(all_targets), test_size=0.2, random_state=42)

    model = GRUModel(input_size=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = ExponentialLR(optimizer, gamma=0.97)

    best_val_loss = float('inf')
    for epoch in range(100):
        model.train()
        losses = []
        loader = DataLoader(WalmartDataset(X_train, y_train), batch_size=32, shuffle=True)
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            mse_loss = nn.MSELoss()(y_pred, y_batch)
            trend_loss = torch.mean((torch.sign(y_pred[1:] - y_pred[:-1]) - torch.sign(y_batch[1:] - y_batch[:-1]))**2)
            loss = mse_loss + 0.1 * trend_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
            val_preds = model(X_val_tensor).squeeze()
            val_loss = nn.MSELoss()(val_preds, y_val_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")
            with open(f"{checkpoint_dir}/input_scaler.pkl", 'wb') as f:
                pickle.dump(input_scaler, f)
            with open(f"{checkpoint_dir}/target_scaler.pkl", 'wb') as f:
                pickle.dump(target_scaler, f)

    return model, input_scaler, target_scaler, X_val, y_val

# === Chạy training ===
df = pd.read_csv("walmart_processed_by_week.csv")
model, input_scaler, target_scaler, X_val, y_val = train_model(df)

# Đánh giá
model.load_state_dict(torch.load(f"{checkpoint_dir}/best_model.pth", map_location=device))
model.eval()
with torch.no_grad():
    val_preds = model(torch.tensor(X_val, dtype=torch.float32).to(device)).squeeze().cpu().numpy()
    val_true = y_val
val_preds_inv = np.expm1(target_scaler.inverse_transform(val_preds.reshape(-1, 1)))
val_true_inv = np.expm1(target_scaler.inverse_transform(val_true.reshape(-1, 1)))
print("== Đánh giá mô hình xu hướng ==")
print("- RMSE:", np.sqrt(mean_squared_error(val_true_inv, val_preds_inv)))
print("- MAE:", mean_absolute_error(val_true_inv, val_preds_inv))
print("- R^2:", r2_score(val_true_inv, val_preds_inv))
# === VẼ BIỂU ĐỒ ===
plt.figure(figsize=(12, 5))
plt.plot(val_true_inv[:len(val_true_inv)//5], label='Actual', marker='o')
plt.plot(val_preds_inv[:len(val_preds_inv)//5], label='Predicted', marker='x')
plt.title("🔍 So sánh GRU: Thực tế vs Dự đoán (20% đầu tập validation)")
plt.xlabel("Sample Index")
plt.ylabel("Weekly Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gru_prediction_vs_actual.png")  # Lưu file ảnh
plt.show()

# === FASTAPI ===
app = FastAPI()

class WeeklyInput(BaseModel):
    data: List[List[float]]

@app.post("/predict")
def predict_sales(input_data: WeeklyInput):
    try:
        if len(input_data.data) != 10:
            raise HTTPException(status_code=400, detail="❌ Cần đúng 10 tuần dữ liệu (10 dòng).")

        # Tạo DataFrame từ input
        input_df = pd.DataFrame(input_data.data, columns=[
            'Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
            'CPI', 'Unemployment', 'WeekOfYear', 'Month'
        ])

        # Tính slope_3w như lúc train
        input_df['slope_3w'] = input_df['Weekly_Sales'].diff().rolling(window=3).mean().fillna(0)

        # Chuẩn hóa các đặc trưng (trừ Weekly_Sales)
        scaled_inputs_only = input_scaler.transform(input_df.iloc[:, 1:])  # gồm cả slope_3w

        # Chuẩn hóa Weekly_Sales (dùng log1p + scaler)
        scaled_weekly_sales = target_scaler.transform(np.log1p(input_df[['Weekly_Sales']]))

        # Ghép lại input đúng thứ tự: Weekly_Sales + 8 đặc trưng còn lại
        scaled_input = np.concatenate([scaled_weekly_sales, scaled_inputs_only], axis=1)

        # Chuyển thành tensor
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)

        # Dự đoán
        model.load_state_dict(torch.load(f"{checkpoint_dir}/best_model.pth", map_location=device))
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(input_tensor).detach().cpu().numpy()

        # Đưa về giá trị thật
        y_pred_real = np.expm1(target_scaler.inverse_transform(y_pred_scaled)[0, 0])

        return {"predicted_weekly_sales": round(float(y_pred_real), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
     uvicorn.run("fixdeepwalmart:app", host="0.0.0.0", port=8000, reload=True)
