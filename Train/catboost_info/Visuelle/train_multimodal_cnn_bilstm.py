# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import matplotlib.pyplot as plt

# # ==================== DATASET ====================
# class SalesDataset(torch.utils.data.Dataset):
#     def __init__(self, df, time_steps=12):
#         self.time_steps = time_steps

#         # Metadata (categorical + numeric)
#         cat_cols = ['season', 'category', 'color', 'fabric']
#         self.cat_encoders = {col: LabelEncoder().fit(df[col]) for col in cat_cols}
#         for col in cat_cols:
#           df.loc[:, col] = self.cat_encoders[col].transform(df[col])


#         numeric_cols = ['external_code', 'retail', 'restock']
#         self.scaler = StandardScaler()
#         df.loc[:, numeric_cols] = self.scaler.fit_transform(df[numeric_cols])


#         self.meta_feats = df[cat_cols + numeric_cols].values.astype(np.float32)
#         self.img_feats = df[[f'img_feat_{i}' for i in range(512)]].values.astype(np.float32)
#         self.release_date = pd.to_datetime(df['release_date']).astype(np.int64) // 10**9
#         self.release_date = StandardScaler().fit_transform(self.release_date.values.reshape(-1, 1)).astype(np.float32)

#         # Combine meta + release_date
#         self.meta_feats = np.concatenate([self.meta_feats, self.release_date], axis=1)

#         self.targets = df[[str(i) for i in range(time_steps)]].values.astype(np.float32)

#     def __len__(self):
#         return len(self.targets)

#     def __getitem__(self, idx):
#         return (
#             torch.tensor(self.img_feats[idx]),
#             torch.tensor(self.meta_feats[idx]),
#             torch.tensor(self.targets[idx])
#         )

# # ==================== MODEL ====================
# class CNNBiLSTMModel(nn.Module):
#     def __init__(self, img_feat_dim=512, meta_feat_dim=8, hidden_dim=128, output_dim=12):
#         super().__init__()
#         self.img_fc = nn.Linear(img_feat_dim, 256)
#         self.meta_fc = nn.Linear(meta_feat_dim, 64)
#         self.bilstm = nn.LSTM(input_size=256 + 64, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
#         self.output = nn.Linear(hidden_dim * 2, output_dim)

#     def forward(self, img_feat, meta_feat):
#         img_emb = torch.relu(self.img_fc(img_feat))  # (B, 256)
#         meta_emb = torch.relu(self.meta_fc(meta_feat))  # (B, 64)

#         x = torch.cat([img_emb, meta_emb], dim=1).unsqueeze(1)  # (B, 1, 320)
#         lstm_out, _ = self.bilstm(x)  # (B, 1, hidden*2)
#         out = self.output(lstm_out.squeeze(1))  # (B, 12)
#         return out

# # ==================== TRAIN FUNCTION ====================
# def train(model, train_loader, val_loader, epochs=30, lr=1e-3, device='cpu'):
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for img_feat, meta_feat, target in train_loader:
#             img_feat, meta_feat, target = img_feat.to(device), meta_feat.to(device), target.to(device)

#             optimizer.zero_grad()
#             pred = model(img_feat, meta_feat)
#             loss = criterion(pred, target)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss / len(train_loader):.4f}")

#     # Eval on validation
#     model.eval()
#     all_preds, all_targets = [], []
#     with torch.no_grad():
#         for img_feat, meta_feat, target in val_loader:
#             img_feat, meta_feat = img_feat.to(device), meta_feat.to(device)
#             pred = model(img_feat, meta_feat)
#             all_preds.append(pred.cpu().numpy())
#             all_targets.append(target.numpy())

#     y_pred = np.concatenate(all_preds, axis=0)
#     y_true = np.concatenate(all_targets, axis=0)

#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae = mean_absolute_error(y_true, y_pred)
#     print(f"\n📊 Validation RMSE: {rmse:.4f} | MAE: {mae:.4f}")

#     # Vẽ biểu đồ
#     plt.figure(figsize=(10, 5))
#     plt.plot(y_true[:10].flatten(), label='Actual', marker='o')
#     plt.plot(y_pred[:10].flatten(), label='Predicted', marker='x')
#     plt.title("Dự đoán doanh thu 12 tháng (mẫu 10 sản phẩm đầu)")
#     plt.xlabel("Tháng")
#     plt.ylabel("Doanh thu (normalized)")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # ==================== MAIN ====================
# if __name__ == "__main__":
#     df = pd.read_csv("store_train_with_image_features.csv")

#     # Train/Val split
#     train_df = df.sample(frac=0.8, random_state=42)
#     val_df = df[~df.index.isin(train_df.index)]

#     train_dataset = SalesDataset(train_df)
#     val_dataset = SalesDataset(val_df)

#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#     model = CNNBiLSTMModel()
#     train(model, train_loader, val_loader, epochs=20, device='cuda' if torch.cuda.is_available() else 'cpu')


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ========================= 1. Dataset =========================
class MultimodalDataset(Dataset):
    def __init__(self, df):
        for col in ['retail', 'season', 'category', 'color', 'fabric']:
            if df[col].dtype == 'object' or str(df[col].dtype).startswith('category'):
                df[col] = df[col].astype('category').cat.codes
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)

        self.img_feats = torch.tensor(df[[f'img_feat_{i}' for i in range(512)]].values, dtype=torch.float32)
        meta_cols = ['retail', 'season', 'category', 'color', 'fabric', 'restock', 'early_customer_count', 'price'] + [f'discount_{i}' for i in range(12)]
        self.meta_feats = torch.tensor(df[meta_cols].values, dtype=torch.float32)
        self.trend_weather_feats = torch.tensor(
            df[[col for col in df.columns if 'trend_' in col or 'weather_' in col]].values,
            dtype=torch.float32
        )
        self.targets = torch.tensor(df[[str(i) for i in range(12)]].values, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.img_feats[idx], self.meta_feats[idx], self.trend_weather_feats[idx], self.targets[idx]


# ========================= 2. Mô hình =========================
class CNNBiLSTMModel(nn.Module):
    def __init__(self, img_feat_dim=512, meta_dim=6 + 1 + 1 + 12, trend_weather_dim=60, hidden_dim=128, output_dim=12):
        super(CNNBiLSTMModel, self).__init__()
        self.img_proj = nn.Linear(img_feat_dim, hidden_dim)
        self.meta_proj = nn.Linear(meta_dim, hidden_dim // 2)
        self.tw_proj = nn.Linear(trend_weather_dim, hidden_dim // 2)
        self.lstm = nn.LSTM(input_size=hidden_dim * 2, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, img_feat, meta_feat, tw_feat):
        img_out = self.img_proj(img_feat)
        meta_out = self.meta_proj(meta_feat)
        tw_out = self.tw_proj(tw_feat)
        combined = torch.cat([img_out, meta_out, tw_out], dim=1).unsqueeze(1)
        lstm_out, _ = self.lstm(combined)
        out = self.fc(lstm_out.squeeze(1))
        return out


# ========================= 3. Load & Merge dữ liệu =========================
df = pd.read_csv('store_train_with_image_features.csv')
trend = pd.read_csv(r"E:\visuelle2\visuelle2\vis2_gtrends_data.csv")
weather = pd.read_csv(r"E:\visuelle2\visuelle2\vis2_weather_data.csv")
restocks = pd.read_csv(r"E:\visuelle2\visuelle2\restocks.csv")
price_discount = pd.read_csv(r"E:\visuelle2\visuelle2\price_discount_series.csv")
customer_data = pd.read_csv(r"E:\visuelle2\visuelle2\customer_data.csv")
print("\n✅ Phân phối doanh thu 12 tháng:")
for i in range(12):
    month_data = df[str(i)]
    print(f"Tháng {i+1}: min={month_data.min()}, max={month_data.max()}, zeros={(month_data==0).sum()} / {len(month_data)}")

# Xử lý thời gian
for col in ['release_date', 'data']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
df['merge_date'] = df['release_date']

trend['date'] = pd.to_datetime(trend['date'], dayfirst=True, errors='coerce')
weather['date'] = pd.to_datetime(weather['date'], dayfirst=True, errors='coerce')

trend = trend.dropna(subset=['date'])
weather = weather.dropna(subset=['date'])

df = df.dropna(subset=['merge_date'])  # đảm bảo bên trái cũng không có null

# Merge trend và weather theo thời gian
df = pd.merge_asof(df.sort_values('merge_date'), trend.sort_values('date'), left_on='merge_date', right_on='date', direction='backward')
df = pd.merge_asof(df.sort_values('merge_date'), weather.sort_values('date'), left_on='merge_date', right_on='date', direction='backward')
print("✅ Sau khi merge trend và weather:")
print("Cột trend/weather:", [col for col in df.columns if 'trend_' in col or 'weather_' in col])
print("Số lượng NULL sau merge:")
print(df[[col for col in df.columns if 'trend_' in col or 'weather_' in col]].isna().sum())
print("Kích thước DataFrame sau merge:", df.shape)
print("✅ Trend/Weather Columns:", [col for col in df.columns if 'trend' in col or 'weather' in col])

# Merge discount series
discount_cols = [str(i) for i in range(12)]
price_discount.rename(columns={col: f'discount_{col}' for col in discount_cols}, inplace=True)
df = df.merge(price_discount, on=['external_code', 'retail'], how='left')

# Merge restocks: lấy tổng số lượng restock trong 4 tuần đầu sau release
restocks['week_date'] = pd.to_datetime(restocks['year'].astype(str) + '-' + restocks['week'].astype(str) + '-1', errors='coerce')
df = df.merge(restocks, on=['external_code', 'retail'], how='left')
df['restock'] = df['qty'].fillna(0)

# Merge customer data: đếm lượt mua theo external_code + retail
customer_data['data'] = pd.to_datetime(customer_data['data'], errors='coerce')
customer_summary = customer_data.groupby(['external_code', 'retail']).size().reset_index()
customer_summary.columns = ['external_code', 'retail', 'early_customer_count']

df = df.merge(customer_summary, on=['external_code', 'retail'], how='left')
df['early_customer_count'] = df['early_customer_count'].fillna(0)

# Dọn dẹp dữ liệu
df.fillna(0, inplace=True)

# ========================= 4. Train / Val =========================
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_loader = DataLoader(MultimodalDataset(train_df), batch_size=64, shuffle=True)
val_loader = DataLoader(MultimodalDataset(val_df), batch_size=64)

# ========================= 5. Huấn luyện =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tw_dim = len([col for col in df.columns if 'trend_' in col or 'weather_' in col])
meta_dim = 6 + 1 + 1 + 12
model = CNNBiLSTMModel(meta_dim=meta_dim, trend_weather_dim=tw_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(20):
    model.train()
    total_loss = 0
    for i, (img, meta, tw, target) in enumerate(train_loader):
        img, meta, tw, target = img.to(device), meta.to(device), tw.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(img, meta, tw)

        if epoch == 0 and i == 0:
            print("\n✅ Batch 1 - Output:", output[0].detach().cpu().numpy())
            print("✅ Batch 1 - Target:", target[0].cpu().numpy())

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")


# ========================= 6. Đánh giá =========================
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for img, meta, tw, target in val_loader:
        img, meta, tw = img.to(device), meta.to(device), tw.to(device)
        output = model(img, meta, tw)
        y_true.append(target.numpy())
        y_pred.append(output.cpu().numpy())

y_true = np.concatenate(y_true, axis=0)
y_pred = np.concatenate(y_pred, axis=0)
print("\n✅ Validation Prediction Summary:")
print("y_true shape:", y_true.shape)
print("y_pred shape:", y_pred.shape)
print("Số lượng giá trị 0 thực tế:", np.sum(y_true == 0))
print("Số lượng giá trị 0 được dự đoán:", np.sum(np.isclose(y_pred, 0, atol=1e-2)))

zero_mask = (y_true == 0)
print("MAE tại điểm zero-sales:", mean_absolute_error(y_true[zero_mask], y_pred[zero_mask]))

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation MAE: {mae:.4f}")

# ========================= 7. Biểu đồ =========================
plt.figure(figsize=(12, 6))
plt.plot(y_true[0], label='Actual', marker='o')
plt.plot(y_pred[0], label='Predicted', marker='x')
plt.title("Dự đoán doanh thu 12 tháng - Mẫu đầu tiên")
plt.xlabel("Tháng")
plt.ylabel("Doanh thu")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
# ========================= 8. Feature Importance (Permutation) =========================
def permutation_feature_importance(model, dataset, feature_type, device='cpu'):
    """
    feature_type: one of ['img', 'meta', 'trend']
    """
    model.eval()
    base_preds = []
    true_vals = []

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for img, meta, tw, target in loader:
            img, meta, tw = img.to(device), meta.to(device), tw.to(device)
            out = model(img, meta, tw)
            base_preds.append(out.cpu().numpy())
            true_vals.append(target.numpy())
    base_preds = np.concatenate(base_preds, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    base_rmse = np.sqrt(mean_squared_error(true_vals, base_preds))

    # Chọn đúng feature block
    if feature_type == 'meta':
        base_feat = dataset.meta_feats.clone()
        num_feats = base_feat.shape[1]
    elif feature_type == 'trend':
        base_feat = dataset.trend_weather_feats.clone()
        num_feats = base_feat.shape[1]
    elif feature_type == 'img':
        base_feat = dataset.img_feats.clone()
        num_feats = base_feat.shape[1]
    else:
        raise ValueError("Invalid feature_type")

    importances = []
    for i in range(num_feats):
        temp_dataset = dataset
        temp_feat = base_feat.clone()
        temp_feat[:, i] = temp_feat[torch.randperm(len(temp_feat)), i]  # shuffle column

        if feature_type == 'meta':
            temp_dataset.meta_feats = temp_feat
        elif feature_type == 'trend':
            temp_dataset.trend_weather_feats = temp_feat
        elif feature_type == 'img':
            temp_dataset.img_feats = temp_feat

        temp_loader = DataLoader(temp_dataset, batch_size=64, shuffle=False)
        perturbed_preds = []
        with torch.no_grad():
            for img, meta, tw, _ in temp_loader:
                img, meta, tw = img.to(device), meta.to(device), tw.to(device)
                out = model(img, meta, tw)
                perturbed_preds.append(out.cpu().numpy())
        perturbed_preds = np.concatenate(perturbed_preds, axis=0)
        rmse = np.sqrt(mean_squared_error(true_vals, perturbed_preds))
        importances.append(rmse - base_rmse)

    return importances
# Meta features
meta_features = ['retail', 'season', 'category', 'color', 'fabric', 'restock', 'early_customer_count', 'price'] + [f'discount_{i}' for i in range(12)]
meta_importance = permutation_feature_importance(model, val_loader.dataset, 'meta', device=str(device))

plt.figure(figsize=(10, 6))
plt.barh(meta_features, meta_importance)
plt.xlabel("RMSE Increase After Permutation")
plt.title("🔍 Feature Importance - Metadata Features")
plt.tight_layout()
plt.grid(True)
plt.show()

# Trend + weather features
tw_features = [col for col in df.columns if 'trend_' in col or 'weather_' in col]
tw_importance = permutation_feature_importance(model, val_loader.dataset, 'trend', device=str(device))
print("\n✅ Feature Importance - Trend/Weather:")
for name, val in zip(tw_features, tw_importance):
    print(f"{name}: RMSE increase = {val:.5f}")

plt.figure(figsize=(12, 8))
plt.barh(tw_features, tw_importance)
plt.xlabel("RMSE Increase After Permutation")
plt.title("🌤️ Feature Importance - Trend & Weather Features")
plt.tight_layout()
plt.grid(True)
plt.show()
# Log min/max/mean của y_pred
print("y_pred stats:", np.min(y_pred), np.max(y_pred), np.mean(y_pred))

# So sánh chênh lệch giữa y_pred và y_true với điều kiện:
print("Top 10 lỗi lớn:")
error = np.abs(y_true - y_pred)
indices = np.argsort(error.sum(axis=1))[-10:]
print("y_pred (bad):", y_pred[indices])
print("y_true (bad):", y_true[indices])
