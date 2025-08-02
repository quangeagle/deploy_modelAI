import sys
import subprocess
import importlib
import joblib
def install_and_import(package):
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    finally:
        globals()[package] = importlib.import_module(package)

# Cài đặt các thư viện cần thiết
for pkg in ['pandas', 'numpy', 'matplotlib', 'sklearn', 'xgboost', 'lightgbm', 'catboost']:
    install_and_import(pkg)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Đọc dữ liệu
file_path = '../Walmart_Sales.csv'
df = pd.read_csv(file_path)

# Tiền xử lý dữ liệu
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date')

# Tạo các features từ Date
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['WeekOfYear'] = df['Date'].dt.isocalendar().week

# Kiểm tra các cột
print('Các cột dữ liệu:', df.columns.tolist())

# Kiểm tra thiếu dữ liệu
print('Thiếu dữ liệu mỗi cột:\n', df.isnull().sum())
df = df.dropna()

# Encode Holiday_Flag nếu chưa phải số
if df['Holiday_Flag'].dtype != np.int64 and df['Holiday_Flag'].dtype != np.float64:
    df['Holiday_Flag'] = df['Holiday_Flag'].astype(int)

# Nếu có cột Promotion thì encode, nếu không thì bỏ qua
if 'Promotion' in df.columns:
    if df['Promotion'].dtype != np.int64 and df['Promotion'].dtype != np.float64:
        df['Promotion'] = df['Promotion'].astype(int)

# Chọn features và target (loại bỏ Date, thêm các features thời gian)
features = ['Store', 'Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
if 'Promotion' in df.columns:
    features.append('Promotion')
X = df[features]
y = df['Weekly_Sales']

# Chia train/test theo thời gian (80% train, 20% test)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Huấn luyện các mô hình
models = {
    'XGBoost': XGBRegressor(random_state=42, n_estimators=100),
    'LightGBM': LGBMRegressor(random_state=42, n_estimators=100),
    'CatBoost': CatBoostRegressor(random_state=42, verbose=0, n_estimators=100)
}
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {'rmse': rmse, 'r2': r2, 'y_pred': y_pred}
    print(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.3f}")
    # Lưu mô hình sau khi huấn luyện
    model_filename = f'{name.lower()}_model.pkl'
    joblib.dump(model, model_filename)
    print(f"  ➤ Đã lưu mô hình {name} vào file {model_filename}")

# Vẽ biểu đồ dự đoán mẫu (20 tuần cuối)
plt.figure(figsize=(15, 6))
plt.plot(y_test.values[-20:], label='Thực tế', marker='o')
for name in models:
    plt.plot(results[name]['y_pred'][-20:], label=f'Dự đoán {name}', marker='x')
plt.title('So sánh dự đoán doanh thu tuần (20 tuần cuối)')
plt.xlabel('Tuần')
plt.ylabel('Doanh thu')
plt.legend()
plt.tight_layout()
plt.show()

# Phân tích ảnh hưởng Holiday_Flag và feature importance
for name, model in models.items():
    print(f"\n{name} Feature Importances:")
    if hasattr(model, 'feature_importances_'):
        for feat, imp in zip(features, model.feature_importances_):
            print(f"  {feat}: {imp:.3f}")
        # Vẽ biểu đồ feature importance
        plt.figure(figsize=(8,4))
        sorted_idx = np.argsort(model.feature_importances_)[::-1]
        plt.bar([features[i] for i in sorted_idx], model.feature_importances_[sorted_idx])
        plt.title(f'Feature Importance - {name}')
        plt.ylabel('Importance')
        plt.xlabel('Feature')
        plt.tight_layout()
        plt.show()
    else:
        print('  Không có feature_importances_')

# Phân tích sâu từng yếu tố ảnh hưởng đến doanh thu
import seaborn as sns
for feat in features:
    plt.figure(figsize=(8,4))
    if df[feat].nunique() <= 10 or df[feat].dtype == int:
        sns.boxplot(x=df[feat], y=df['Weekly_Sales'])
        plt.title(f'Ảnh hưởng của {feat} đến doanh thu (boxplot)')
        plt.xlabel(feat)
        plt.ylabel('Doanh thu tuần')
    else:
        sns.scatterplot(x=df[feat], y=df['Weekly_Sales'], alpha=0.3)
        plt.title(f'Quan hệ giữa {feat} và doanh thu (scatter)')
        plt.xlabel(feat)
        plt.ylabel('Doanh thu tuần')
    plt.tight_layout()
    plt.show()

# Nhận xét tổng quan về các yếu tố quan trọng nhất dựa trên XGBoost
xgb_importances = models['XGBoost'].feature_importances_
imp_sorted_idx = np.argsort(xgb_importances)[::-1]
print("\nCác yếu tố quan trọng nhất theo XGBoost:")
for i in imp_sorted_idx:
    print(f"  {features[i]}: {xgb_importances[i]:.3f}")
print("\nBạn có thể dựa vào các biểu đồ và bảng trên để nhận biết yếu tố nào ảnh hưởng mạnh nhất đến doanh thu tuần của Walmart.")

# Phân tích ảnh hưởng ngày lễ
sns.boxplot(x=df['Holiday_Flag'], y=df['Weekly_Sales'])
plt.title('Ảnh hưởng của ngày lễ đến doanh thu')
plt.xlabel('Holiday_Flag (0: Không, 1: Có)')
plt.ylabel('Doanh thu tuần')
plt.show()

# Nếu có Promotion thì vẽ thêm
if 'Promotion' in df.columns:
    sns.boxplot(x=df['Promotion'], y=df['Weekly_Sales'])
    plt.title('Ảnh hưởng của khuyến mãi đến doanh thu')
    plt.xlabel('Promotion (0: Không, 1: Có)')
    plt.ylabel('Doanh thu tuần')
    plt.show()
