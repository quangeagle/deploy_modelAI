import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Đọc dữ liệu
file_path = "extended_ecommerce_sales_forecast_dataset.csv"
df = pd.read_csv(file_path)

target_col = "Sales Forecast"

# 1. Phân tích dữ liệu
print("\n--- Thông tin tổng quan ---")
print(df.info())
print("\n--- Thống kê mô tả ---")
print(df.describe())
print("\n--- Số lượng giá trị thiếu ---")
print(df.isnull().sum())

# Vẽ phân phối biến mục tiêu
plt.figure(figsize=(6,4))
df[target_col].hist(bins=30)
plt.title('Phân phối Sales Forecast')
plt.show()

# Vẽ phân phối các đặc trưng số
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    if col != target_col:
        plt.figure(figsize=(6,4))
        df[col].hist(bins=30)
        plt.title(f'Phân phối {col}')
        plt.show()

# Vẽ ma trận tương quan
plt.figure(figsize=(12,8))
corr = df.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Ma trận tương quan")
plt.show()

# 2. Loại bỏ outlier bằng IQR cho tất cả cột số (trừ cột mục tiêu)
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    filtered_df = df[(df[col] >= lower) & (df[col] <= upper)]
    return filtered_df

for col in df.select_dtypes(include=['float64', 'int64']).columns:
    if col != target_col:
        df = remove_outliers_iqr(df, col)

# Ensure df is still a DataFrame after outlier removal
assert isinstance(df, pd.DataFrame)

print("\n--- Sau khi loại outlier ---")
print(df[target_col].describe())
plt.figure(figsize=(6,4))
df[target_col].hist(bins=30)
plt.title('Phân phối Sales Forecast sau khi loại outlier')
plt.show()

# 3. Mã hóa tất cả các cột object (chuỗi) thành số, trừ cột mục tiêu
for col in df.columns:
    if df[col].dtype == "object" and col != target_col:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# --- Feature Engineering: tạo feature mới ---
# Tạo feature tương tác
df['Shipping_Customer'] = df['Shipping Speed'] * df['Customer Behavior']
df['Engagement_AdSpend'] = df['Engagement Rate'] * df['Advertising Spend']
df['Discount_AvgOrder'] = df['Discount Rate'] * df['Average Order Value']

# Tạo feature phi tuyến
import numpy as np
# (đã import ở đầu file, nhưng để chắc chắn)
df['Log_AvgOrder'] = np.log1p(df['Average Order Value'])
df['Square_Shipping'] = df['Shipping Speed'] ** 2

# In lại ma trận tương quan mới với Sales Forecast
corr_new = df.corr()
print("\n--- Ma trận tương quan mới với feature mới ---")
print(pd.Series(corr_new['Sales Forecast']).sort_values())

# 4. Train/test Random Forest và Linear Regression
X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

print_metrics("Random Forest", y_test, y_pred_rf)
print_metrics("Linear Regression", y_test, y_pred_lr)

# 5. Vẽ feature importance cho Random Forest
importances = rf_model.feature_importances_
features = list(X.columns)
indices = importances.argsort()[::-1]

plt.figure(figsize=(12,6))
plt.title("Feature Importance đối với Sales Forecast (Random Forest)")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# 6. Vẽ so sánh doanh thu thực tế và dự đoán (50 mẫu đầu)
plt.figure(figsize=(10,6))
plt.plot(list(y_test)[:50], label="Thực tế", marker='o')
plt.plot(list(y_pred_rf)[:50], label="Random Forest", marker='x')
plt.plot(list(y_pred_lr)[:50], label="Linear Regression", marker='s')
plt.title("So sánh doanh thu thực tế và dự đoán (50 mẫu đầu)")
plt.xlabel("Mẫu")
plt.ylabel("Sales Forecast")
plt.legend()
plt.tight_layout()
plt.show()