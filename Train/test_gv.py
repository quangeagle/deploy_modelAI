# 📦 Thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn import __version__ as sklearn_version
# 📂 Đọc dữ liệu
df = pd.read_excel("E:/TrainAI/Train/test_gv.xlsx", engine="openpyxl")
from sklearn.metrics import r2_score
print("Các cột trong file Excel:", df.columns.tolist())
# ✅ Xử lý ngày tháng
df['Ngày đăng'] = pd.to_datetime(df['Ngày đăng'], format="%d/%m")
df['Tháng đăng'] = df['Ngày đăng'].dt.month
df['Ngày đăng'] = df['Ngày đăng'].dt.day

# ✅ Encode các cột text (dạng ngắn)
categorical_cols = ['Ngành hàng', 'Gần lễ gì', 'Trend', 'Tỉnh thành', 'Thời tiết']
if sklearn_version >= "1.2":
    encoder = OneHotEncoder(sparse_output=False)
else:
    encoder = OneHotEncoder(sparse=False)
encoded_cat = encoder.fit_transform(df[categorical_cols])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_cols))

# ✅ Trích đặc trưng từ tên SP và mô tả (rút gọn bằng độ dài chuỗi)
df['Tên SP len'] = df['Tên SP'].apply(len)
df['Mô tả len'] = df['Mô tả'].apply(len)

# ✅ Tạo dataframe đặc trưng đầu vào
X = pd.concat([
    df[['Giá (k)', 'Ngày đăng', 'Tháng đăng', 'Tên SP len', 'Mô tả len']],
    encoded_cat_df
], axis=1)

# 🎯 Nhãn đầu ra
y = df['Dự báo doanh thu (triệu)']


# 🔀 Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Train XGBoost
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# 🎯 Dự đoán
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
# 📊 Đánh giá
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
print("R² score:", r2)