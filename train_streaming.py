from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Streaming data
print("Bắt đầu load dataset streaming...")
dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K", split="train", streaming=True)

# 2. Feature mapping
feature_names = [
    "city_id", "store_id", "management_group_id",
    "first_category_id", "second_category_id", "third_category_id",
    "product_id", "stock_hour6_22_cnt", "discount", "holiday_flag",
    "activity_flag", "precpt", "avg_temperature", "avg_humidity", "avg_wind_level"
]

def preprocess(ex):
    dt = ex["dt"]
    try:
        dt_obj = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        dt_obj = datetime.strptime(dt, "%Y-%m-%d")
    hour = getattr(dt_obj, "hour", 0)
    dow = dt_obj.weekday()
    # Lấy các feature số/categorical
    feats = [ex[f] for f in feature_names]
    feats = [float(x) for x in feats]  # Đảm bảo kiểu float
    feats = [hour, dow] + feats
    return {"X": feats, "y": ex["sale_amount"]}

print("Đang map features...")
dataset = dataset.map(preprocess)

# 3. Batch generator
def batch_gen(ds, batch_size=1024):
    Xb, yb = [], []
    for ex in ds:
        Xb.append(ex["X"]); yb.append(ex["y"])
        if len(Xb)==batch_size:
            yield np.array(Xb), np.array(yb)
            Xb, yb = [], []
    if Xb:
        yield np.array(Xb), np.array(yb)

# 4. Initialize model
print("Khởi tạo mô hình RandomForest...")
model = RandomForestRegressor(n_estimators=50, warm_start=True, random_state=42)

# 5. Train in streaming batches
print("Lấy batch đầu tiên để train...")
bgen = batch_gen(dataset, batch_size=2048)
X0, y0 = next(bgen)
print(f"Đã lấy batch đầu: {X0.shape}, bắt đầu train...")
model.fit(X0, y0)
for i, (Xb, yb) in enumerate(bgen):
    print(f"Train batch tiếp theo số {i+2} với shape {Xb.shape}...")
    model.n_estimators += 10
    model.fit(Xb, yb)

# 6. Load one more batch for test
print("Lấy batch test...")
Xtest, ytest = next(batch_gen(dataset, batch_size=2048))
print(f"Batch test shape: {Xtest.shape}")
print("Predicting...")
ypred = model.predict(Xtest)

# Tính các chỉ số đánh giá
mae = mean_absolute_error(ytest, ypred)
rmse = np.sqrt(mean_squared_error(ytest, ypred))
r2 = r2_score(ytest, ypred)
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# 7. Plot compare
print("Vẽ biểu đồ kết quả...")
plt.figure(figsize=(10,5))
plt.plot(ytest[:200], label="Actual", marker='.')
plt.plot(ypred[:200], label="Predicted", marker='x')
plt.legend()
plt.title("Actual vs Predicted Sales (sample)")
plt.xlabel("Record index"); plt.ylabel("Sales")
plt.show() 