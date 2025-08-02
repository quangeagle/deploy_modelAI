import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from pandas import DataFrame
import gc 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

def save_checkpoint(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_checkpoint(path):
         import pickle
         try:
             with open(path, "rb") as f:
                 return pickle.load(f)
         except Exception as e:
             print(f"Error loading checkpoint: {e}")
             raise


if os.path.exists("checkpoint_step1.pkl"):
    print("[1/9] Đã load dữ liệu từ checkpoint.")
    train, items, stores, holidays, transactions, oil = load_checkpoint("checkpoint_step1.pkl")
else:
    print("[1/9] Đang load dữ liệu gốc...")
    train = pd.read_csv("train.csv", parse_dates=["date"], low_memory=False)
    items = pd.read_csv("items.csv")
    stores = pd.read_csv("stores.csv")
    holidays = pd.read_csv("holidays_events.csv", parse_dates=["date"])
    transactions = pd.read_csv("transactions.csv", parse_dates=["date"])
    oil = pd.read_csv("oil.csv", parse_dates=["date"])
    save_checkpoint((train, items, stores, holidays, transactions, oil), "checkpoint_step1.pkl")


if os.path.exists("checkpoint_step2.pkl"):
    print("[2/9] Đã load dữ liệu gộp từ checkpoint.")
    train = load_checkpoint("checkpoint_step2.pkl")
else:
    print("[2/9] Đang gộp dữ liệu...")
    train = train.merge(items, on="item_nbr", how="left")
    train = train.merge(stores, on="store_nbr", how="left")
    train = train.merge(transactions, on=["store_nbr", "date"], how="left")
    train = train.merge(oil.rename(columns={"dcoilwtico": "oil_price"}), on="date", how="left")
    train["oil_price"].fillna(method="ffill", inplace=True)  # 👈 Fix lỗi NaN trong oil
    save_checkpoint(train, "checkpoint_step2.pkl")


if "day" not in train.columns:
    print("[3/9] Tạo đặc trưng ngày tháng...")
    train["day"] = train["date"].dt.day
    train["month"] = train["date"].dt.month
    train["dayofweek"] = train["date"].dt.dayofweek
    train["is_weekend"] = train["dayofweek"].isin([5, 6]).astype(int)


if "is_holiday" not in train.columns:
    print("[4/9] Đang xử lý ngày lễ quốc gia...")
    nat_holidays = holidays.loc[holidays["locale"] == "National", ["date"]].drop_duplicates()
    nat_holidays["is_holiday"] = 1
    holiday_dict = dict(zip(nat_holidays["date"], nat_holidays["is_holiday"]))
    train["is_holiday"] = train["date"].map(holiday_dict.get).fillna(0).astype(int)



print("[5/9] Đang encode...")
for col in ["family", "city", "state", "type"]:
    if train[col].dtype == "object":
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))


if "log_unit_sales" not in train.columns:
    print("[6/9] Log-transform mục tiêu...")
    train["unit_sales"] = train["unit_sales"].clip(0, None)
    train["log_unit_sales"] = np.log1p(train["unit_sales"])

# [7/9] Tách tập
# [7/9] Tách tập train/valid và tiết kiệm bộ nhớ
print("[7/9] Tách tập train/valid...")

split_date = pd.to_datetime("2017-07-01")
features = [
    "store_nbr", "item_nbr", "family", "city", "state", "type",
    "transactions", "oil_price", "day", "month", "dayofweek", "is_weekend", "is_holiday"
]
target = "log_unit_sales"

all_needed_cols = features + [target, "date"]
train = train[all_needed_cols]

# Ép kiểu giảm bộ nhớ
for col in features:
    if train[col].dtype == 'float64':
        train[col] = train[col].astype('float32')
    elif train[col].dtype == 'int64':
        train[col] = train[col].astype('int32')

# Chia tập mà không copy
train_set = train.loc[train["date"] < split_date].dropna(subset=list(features) + [target])
valid_set = train.loc[train["date"] >= split_date].dropna(subset=list(features) + [target])




# Giải phóng bộ nhớ
del train
gc.collect()



# [8/9] Train LightGBM
# [8/9] Train LightGBM an toàn hơn
print("[8/9] Đang train mô hình...")

lgb_train = lgb.Dataset(train_set[features].values, label=train_set[target].values)
lgb_valid = lgb.Dataset(valid_set[features].values, label=valid_set[target].values)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "verbose": -1
}

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50)]
)
with open("lgbm_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Đã lưu mô hình LightGBM.")
# [9/9] Dự đoán và đánh giá
print("[9/9] Đang đánh giá mô hình...")
y_pred_log = model.predict(valid_set[features])
y_pred = np.expm1(np.asarray(y_pred_log))
y_true = np.expm1(valid_set[target])
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"Validation RMSE: {rmse:.4f}")
# Vẽ biểu đồ dự báo vs thực tế
print("🔍 Vẽ biểu đồ dự báo so với thực tế...")
plt.figure(figsize=(10, 5))
plt.plot(y_true.values[:300], label="Thực tế", marker='o')
plt.plot(y_pred[:300], label="Dự báo", marker='x')
plt.title("So sánh Dự báo và Thực tế (300 mẫu đầu)")
plt.xlabel("Mẫu")
plt.ylabel("Unit Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Tính điểm R2
r2 = r2_score(y_true, y_pred)
print(f"Điểm R²: {r2:.4f}")

# Vẽ biểu đồ feature importance
print("📊 Vẽ biểu đồ Feature Importance...")
importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importance()
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis")
plt.title("Độ quan trọng của các đặc trưng (Feature Importance)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
