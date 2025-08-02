import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def prepare_ml_data(df, lookback=10):
    feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
    all_features = []
    all_targets = []
    for store_id in df['Store'].unique():
        store_df = df[df['Store'] == store_id].sort_values('Week_Index')
        data = store_df[feature_cols].values.astype(np.float32)
        for i in range(len(data) - lookback):
            seq = data[i:i+lookback].flatten()
            target = data[i+lookback, 0]
            all_features.append(seq)
            all_targets.append(target)
    return np.array(all_features), np.array(all_targets)

# Load dữ liệu
df = pd.read_csv("walmart_processed_by_week.csv")
X, y = prepare_ml_data(df, lookback=10)

print(f"📊 Target data range: {y.min():,.2f} - {y.max():,.2f}")
print(f"📊 Target mean: {y.mean():,.2f}")
print(f"📊 Target std: {y.std():,.2f}")

# Tạo target scaler
target_scaler = StandardScaler()
target_scaler.fit(y.reshape(-1, 1))

print(f"\n🔧 Target Scaler:")
print(f"   Mean: {target_scaler.mean_[0]:,.2f}")
print(f"   Scale: {target_scaler.scale_[0]:,.2f}")

# Test inverse transform
test_scaled = 0.5  # Giá trị scaled
test_real = target_scaler.inverse_transform([[test_scaled]])[0][0]
print(f"\n🧪 Test inverse transform:")
print(f"   Scaled value: {test_scaled}")
print(f"   Real value: ${test_real:,.2f}")

# Lưu target scaler
with open("model_checkpoints/target_scaler.pkl", "wb") as f:
    pickle.dump(target_scaler, f)

print(f"\n✅ Đã lưu target_scaler.pkl") 