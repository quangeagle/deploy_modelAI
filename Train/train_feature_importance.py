# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from xgboost import XGBRegressor, plot_importance
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.preprocessing import StandardScaler

# # 1. Load dataset
# df = pd.read_csv('walmart_processed_by_week.csv')  # Đổi tên file nếu khác

# # 2. Xử lý cột ngày
# df['Date'] = pd.to_datetime(df['Date'])

# # 3. Tạo đặc trưng tuần Black Friday (tuần 47 hoặc 48)
# df['Is_BlackFriday_Week'] = df['WeekOfYear'].apply(lambda x: 1 if x in [47, 48] else 0)

# # 4. Chọn đặc trưng và target
# features = [
#     'Holiday_Flag', 'Temperature', 'Fuel_Price',
#     'CPI', 'Unemployment',
#     'Week_Index', 'WeekOfYear', 'Month', 'Year', 'Is_BlackFriday_Week'
# ]
# target = 'Weekly_Sales'

# X = df[features]
# y = df[target]

# # 5. Chuẩn hóa dữ liệu
# scaler = StandardScaler()
# X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

# # 6. Chia dữ liệu train/test
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42
# )

# # 7. Train mô hình XGBoost
# model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
# model.fit(X_train, y_train)

# # 8. Dự đoán và đánh giá
# y_pred = model.predict(X_test)
# r2 = r2_score(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f'✅ R2 Score: {r2:.4f}')
# print(f'📉 RMSE: {rmse:.2f}')

# # 9. Phân tích tầm quan trọng đặc trưng
# plt.figure(figsize=(10, 6))
# plot_importance(model, importance_type='gain', max_num_features=10)
# plt.title("🎯 Feature Importance (XGBoost)")
# plt.tight_layout()
# plt.show()

# # 10. Tạo bảng so sánh dự đoán trong tuần Black Friday
# df['Predicted'] = model.predict(scaler.transform(X))
# black_friday_df = df[df['Is_BlackFriday_Week'] == 1]
# print("\n📈 Doanh số dự đoán trong tuần Black Friday:")
# print(black_friday_df[['Date', 'Weekly_Sales', 'Predicted']].sort_values('Date'))
# plt.figure(figsize=(12, 5))
# sns.boxplot(x='WeekOfYear', y='Weekly_Sales', data=df)
# plt.title('Doanh số theo tuần trong năm')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(10, 5))
# sns.boxplot(x='Month', y='Weekly_Sales', data=df)
# plt.title('Doanh số theo tháng')
# plt.tight_layout()
# plt.show()
# sns.boxplot(x='Month', y='Weekly_Sales', data=df)
# plt.title('Doanh số theo tháng')
# plt.show()

# # Doanh số theo tuần trong năm
# sns.boxplot(x='WeekOfYear', y='Weekly_Sales', data=df)
# plt.title('Doanh số theo tuần trong năm')
# plt.xticks(rotation=90)
# plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Load dataset
df = pd.read_csv('walmart_processed_by_week.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Is_BlackFriday_Week'] = df['WeekOfYear'].apply(lambda x: 1 if x in [47, 48] else 0)

# 2. Chọn đặc trưng
features = [
    'Holiday_Flag', 'Temperature', 'Fuel_Price',
    'CPI', 'Unemployment',
    'Week_Index', 'WeekOfYear', 'Month', 'Year', 'Is_BlackFriday_Week'
]
target = 'Weekly_Sales'
X = df[features]
y = df[target]

# 3. Chuẩn hóa
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===========================
# 4. Train XGBoost
# ===========================
model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)

r2_xgb = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f'📘 XGBoost R²: {r2_xgb:.4f}, RMSE: {rmse_xgb:.2f}')

# ===========================
# 5. Train GRU (PyTorch)
# ===========================
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# Chuẩn bị dữ liệu GRU
def to_tensor(x): return torch.tensor(x, dtype=torch.float32)

X_train_seq = to_tensor(X_train.values).unsqueeze(1)
X_test_seq = to_tensor(X_test.values).unsqueeze(1)
y_train_tensor = to_tensor(y_train.values).unsqueeze(1)
y_test_tensor = to_tensor(y_test.values).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_seq, y_train_tensor), batch_size=64, shuffle=True)

# Khởi tạo mô hình
gru = GRUNet(input_size=X_train.shape[1], hidden_size=64, output_size=1)
optimizer = torch.optim.Adam(gru.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Train GRU
for epoch in range(50):
    for xb, yb in train_loader:
        pred = gru(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Dự đoán GRU
y_pred_gru = gru(X_test_seq).detach().numpy().flatten()
r2_gru = r2_score(y_test, y_pred_gru)
rmse_gru = np.sqrt(mean_squared_error(y_test, y_pred_gru))
print(f'📕 GRU R²: {r2_gru:.4f}, RMSE: {rmse_gru:.2f}')

# ===========================
# 6. So sánh kết quả
# ===========================
plt.bar(['XGBoost', 'GRU'], [r2_xgb, r2_gru], color=['skyblue', 'salmon'])
plt.title('So sánh R² XGBoost vs GRU')
plt.ylabel('R² Score')
plt.ylim(0, 1)
plt.show()

# ===========================
# 7. Phân tích chi tiết XGBoost + biểu đồ tuần/tháng
# ===========================
plt.figure(figsize=(10, 6))
plot_importance(model_xgb, importance_type='gain', max_num_features=10)
plt.title("🎯 Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()

df['Predicted_XGB'] = model_xgb.predict(scaler.transform(X))
df['Predicted_GRU'] = gru(torch.tensor(scaler.transform(X), dtype=torch.float32).unsqueeze(1)).detach().numpy()

black_friday_df = df[df['Is_BlackFriday_Week'] == 1]
print("\n📈 Doanh số dự đoán trong tuần Black Friday:")
print(black_friday_df[['Date', 'Weekly_Sales', 'Predicted_XGB', 'Predicted_GRU']].sort_values('Date'))

# Boxplot tuần & tháng
plt.figure(figsize=(12, 5))
sns.boxplot(x='WeekOfYear', y='Weekly_Sales', data=df)
plt.title('Doanh số theo tuần trong năm')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x='Month', y='Weekly_Sales', data=df)
plt.title('Doanh số theo tháng')
plt.tight_layout()
plt.show()
