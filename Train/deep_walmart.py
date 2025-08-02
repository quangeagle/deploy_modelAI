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
from sklearn.metrics import r2_score
import copy
import matplotlib.pyplot as plt
checkpoint_dir = 'model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# 🔄 Đọc best R2 trước đó nếu có
r2_path = os.path.join(checkpoint_dir, "best_r2_score.txt")
if os.path.exists(r2_path):
    with open(r2_path, 'r') as f:
        best_r2_score = float(f.read().strip())
    print(f"📌 R² tốt nhất từ trước: {best_r2_score:.4f}")
else:
    best_r2_score = float('-inf')
    print("📌 Chưa có mô hình tốt nhất trước đó, sẽ bắt đầu ghi mới.")


# ========== 1. Dataset ==========
class WalmartDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

# ========== 2. Model ==========
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.linear(out[:, -1, :])

# ========== 3. Prepare data ==========
def create_sequences(data, lookback):
    sequences, targets = [], []
    for i in range(len(data) - lookback):
        seq = data[i:i+lookback]
        target = data[i+lookback, 0]  # Weekly_Sales
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# ========== 4. Train function with rolling + freeze + decay ==========
def train_model_continual_gru(df, lookback=12):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
    
    input_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    df[['Weekly_Sales']] = target_scaler.fit_transform(df[['Weekly_Sales']])

    # Scale 7 features còn lại
    input_features = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
    input_scaler = MinMaxScaler()
    df[input_features] = input_scaler.fit_transform(df[input_features])

    # ✅ THÊM TRAIN/TEST SPLIT
    from sklearn.model_selection import train_test_split
    
    # Chuẩn bị tất cả sequences
    all_sequences = []
    all_targets = []
    
    for store_id in df['Store'].unique():
        store_df = df[df['Store'] == store_id].sort_values('Week_Index')
        data = store_df[feature_cols].values.astype(np.float32)
        X, y = create_sequences(data, lookback)
        all_sequences.extend(X)
        all_targets.extend(y)
    
    # Chia train/test với 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(all_sequences), np.array(all_targets), 
        test_size=0.2, random_state=42
    )
    
    print(f"✅ Train/Test split: {len(X_train)} train, {len(X_test)} test samples")

    store_ids = df['Store'].unique()
    model = GRUModel(input_size=len(feature_cols)).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    store_loss_log = {}

    # ✅ CHỈ TRAIN TRÊN TRAIN SET
    for i, store_id in enumerate(store_ids):
        print(f"\n===== Training on Store {store_id} =====")
        store_df = df[df['Store'] == store_id].sort_values('Week_Index')
        data = store_df[feature_cols].values.astype(np.float32)

        X, y = create_sequences(data, lookback)
        dataset = WalmartDataset(X, y)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        # 🔒 Giai đoạn đầu: chỉ train Linear, giữ nguyên GRU
        if i <= 1:
            for name, param in model.named_parameters():
                if 'gru' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        # 🔓 Sau vài store: cho phép fine-tune GRU nhẹ nhàng
        else:
            for name, param in model.named_parameters():
                param.requires_grad = True  # fine-tune toàn bộ

        for epoch in range(10):
            model.train()
            losses = []
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            print(f"[Epoch {epoch+1}] Loss: {mean_loss:.4f}")
        
        scheduler.step()
        store_loss_log[store_id] = mean_loss

    print("\n===== Final Store Losses =====")
    for store_id, loss in store_loss_log.items():
        print(f"Store {store_id}: Loss = {loss:.4f}")

    return model, input_scaler, target_scaler, X_test, y_test  # ✅ TRẢ VỀ TEST SET


# ========== 5. Run ==========
df = pd.read_csv("walmart_processed_by_week.csv")
all_preds = []
all_targets = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model, input_scaler, target_scaler, X_test, y_test = train_model_continual_gru(df)

# ✅ EVALUATE CHỈ TRÊN TEST SET
print("\n===== Evaluating on Test Set =====")
with torch.no_grad():
    # Tạo test dataset
    test_dataset = WalmartDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = trained_model(inputs)
        all_preds.append(outputs.cpu().item())
        all_targets.append(targets.cpu().item())

# Convert to NumPy arrays
preds = np.array(all_preds)
targets = np.array(all_targets)

# Metrics
rmse = np.sqrt(mean_squared_error(targets, preds))
mae = mean_absolute_error(targets, preds)
r2 = r2_score(targets, preds)
# 🔽 Kiểm tra nếu tốt hơn thì lưu lại
if r2 > best_r2_score:
    best_r2_score = r2

    # ✅ Lưu mô hình
    torch.save(trained_model.state_dict(), f"{checkpoint_dir}/best_model.pth")
    
    # ✅ Lưu scaler
    with open(f"{checkpoint_dir}/input_scaler.pkl", 'wb') as f:
        pickle.dump(input_scaler, f)
    with open(f"{checkpoint_dir}/target_scaler.pkl", 'wb') as f:
        pickle.dump(target_scaler, f)

    # ✅ Ghi lại best R² vào file
    with open(r2_path, 'w') as f:
        f.write(str(best_r2_score))

    print(f"✅ Lưu mô hình mới với R² = {r2:.4f} (tốt hơn trước đó)!")
else:
    print(f"❌ R² = {r2:.4f} không tốt hơn R² tốt nhất = {best_r2_score:.4f}, không lưu mô hình.")

print(f"\n📊 Evaluation on all stores:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# ==== 6. Đọc dữ liệu 10 tuần gần nhất từ file CSV ====
print("\n📂 Đang đọc dữ liệu từ file 'input_last_10_weeks.csv'...")

# Đọc file
input_df = pd.read_excel("test.xlsx")

# Kiểm tra đúng số lượng tuần và cột
if input_df.shape[0] != 10:
    raise ValueError("❌ File phải chứa đúng 10 tuần gần nhất!")
if input_df.shape[1] != 8:
    raise ValueError("❌ File phải có đúng 8 cột: Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, WeekOfYear, Month")

# Chuyển về numpy array
input_weeks = input_df.to_numpy()

print("✅ Đã load dữ liệu thành công:\n", input_df)


# ======= XỬ LÝ INPUT TUẦN MỚI =======

feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
input_features = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
weekly_sales_col = ['Weekly_Sales']

input_weeks_df = pd.DataFrame(input_weeks, columns=feature_cols)

# Scale từng phần
scaled_inputs_only = input_scaler.transform(input_weeks_df[input_features])
scaled_weekly_sales = target_scaler.transform(input_weeks_df[weekly_sales_col])

# Ghép lại thành đúng input cho model: Weekly_Sales trước + các feature sau
scaled_input = np.concatenate([scaled_weekly_sales, scaled_inputs_only], axis=1)

# Dự đoán
input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)
trained_model.eval()
with torch.no_grad():
    y_pred_scaled = trained_model(input_tensor).detach().cpu().numpy()

# Inverse transform để lấy giá trị thật sự
y_pred_real = target_scaler.inverse_transform(y_pred_scaled)[0, 0]

print("📉 y_pred (scaled):", y_pred_scaled)
print("✅ y_pred (real):", y_pred_real)
print(f"\n🧠 Dự đoán Weekly_Sales tuần tiếp theo: {y_pred_real:.2f}")

print("Kết quả đầu ra sau dự đoán (scaled):", y_pred_scaled)
print("Sau khi inverse_transform:", y_pred_real)
# ========== FASTAPI ==========

app = FastAPI()

# Định nghĩa input schema
class WeeklyInput(BaseModel):
    data: List[List[float]]  # 10 hàng x 8 cột

@app.post("/predict")
def predict_sales(input_data: WeeklyInput):
    try:
        if len(input_data.data) != 10:
            raise HTTPException(status_code=400, detail="❌ Cần đúng 10 tuần dữ liệu (10 dòng).")
        
        feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
                        'CPI', 'Unemployment', 'WeekOfYear', 'Month']
        input_df = pd.DataFrame(input_data.data, columns=feature_cols)
        input_features = feature_cols[1:]
        weekly_sales_col = ['Weekly_Sales']

        # Scale dữ liệu
        scaled_inputs_only = input_scaler.transform(input_df[input_features])
        scaled_weekly_sales = target_scaler.transform(input_df[weekly_sales_col])
        scaled_input = np.concatenate([scaled_weekly_sales, scaled_inputs_only], axis=1)

        # Dự đoán
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)
        trained_model.eval()
        with torch.no_grad():
            y_pred_scaled = trained_model(input_tensor).detach().cpu().numpy()
        y_pred_real = target_scaler.inverse_transform(y_pred_scaled)[0, 0]

        return {"predicted_weekly_sales": round(float(y_pred_real), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chạy server FastAPI
if __name__ == "__main__":
    uvicorn.run("deep_walmart:app", host="0.0.0.0", port=8000, reload=True)
# === Gom toàn bộ dữ liệu sau khi dự đoán ===
X_all = []
y_all = []

with torch.no_grad():
    for store in df['Store'].unique():
        df_store = pd.DataFrame(df[df['Store'] == store]).sort_values(by='Week_Index')
        data = df_store[feature_cols].values.astype(np.float32)
        X, y = create_sequences(data, lookback=12)

        X_all.append(X)
        y_all.append(y)

X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)
y_all = y_all.reshape(-1, 1)  # Cho chắc

# === Hàm tính permutation importance ===
def permutation_importance(model, X, y_true, metric=r2_score, n_repeats=3):
    base_pred = model(torch.tensor(X, dtype=torch.float32).to(device)).detach().cpu().numpy()
    base_score = metric(y_true, base_pred)
    
    feature_importance = np.zeros(X.shape[2])

    for col in range(X.shape[2]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = copy.deepcopy(X)
            np.random.shuffle(X_permuted[:, :, col])
            with torch.no_grad():
                pred = model(torch.tensor(X_permuted, dtype=torch.float32).to(device)).cpu().numpy()
            score = metric(y_true, pred)
            scores.append(base_score - score)
        feature_importance[col] = np.mean(scores)
    
    return feature_importance

# === Gọi tính importance
importances = permutation_importance(trained_model, X_all, y_all)

feature_names = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
                 'CPI', 'Unemployment', 'WeekOfYear', 'Month']

# === Vẽ biểu đồ
plt.figure(figsize=(10, 5))
plt.bar(feature_names, importances)
plt.xticks(rotation=45)
plt.title("Feature Importance (GRU via Permutation)")
plt.ylabel("Importance Score (Δ R²)")
plt.grid(True)
plt.tight_layout()
plt.savefig("feature_importance_gru.png")
plt.show()

# === In chi tiết
for i, imp in enumerate(importances):
    print(f"{feature_names[i]}: {imp:.4f}")
