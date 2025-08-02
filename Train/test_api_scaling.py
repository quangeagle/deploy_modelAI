import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load dữ liệu gốc để kiểm tra
df = pd.read_csv("walmart_processed_by_week.csv")
print("📊 DỮ LIỆU GỐC:")
print(f"   Weekly_Sales range: {df['Weekly_Sales'].min():,.2f} - {df['Weekly_Sales'].max():,.2f}")
print(f"   Weekly_Sales mean: {df['Weekly_Sales'].mean():,.2f}")
print(f"   Weekly_Sales std: {df['Weekly_Sales'].std():,.2f}")

# Load model và scaler
with open("model_checkpoints/best_ml_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model_checkpoints/ml_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print(f"\n🔧 SCALER INFO:")
print(f"   Scaler mean_: {scaler.mean_[:8]}")  # 8 features đầu tiên
print(f"   Scaler scale_: {scaler.scale_[:8]}")

# Test với dữ liệu thực tế
def prepare_test_data(df, store_id=1, lookback=10):
    """Chuẩn bị test data từ store thực tế"""
    feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
    
    store_df = df[df['Store'] == store_id].sort_values('Week_Index')
    data = store_df[feature_cols].values.astype(np.float32)
    
    # Lấy 10 tuần cuối cùng
    last_10_weeks = data[-lookback:]
    test_sequence = last_10_weeks.flatten()  # 80 features
    
    # Target thực tế (tuần tiếp theo)
    actual_target = data[-1, 0]  # Weekly_Sales của tuần cuối
    
    return test_sequence, actual_target

# Test với store 1
test_sequence, actual_target = prepare_test_data(df, store_id=1)
print(f"\n🧪 TEST SEQUENCE:")
print(f"   Input shape: {test_sequence.shape}")
print(f"   Actual target: ${actual_target:,.2f}")

# Scale input
input_scaled = scaler.transform(test_sequence.reshape(1, -1))
print(f"\n📏 SCALED INPUT:")
print(f"   Scaled input range: {input_scaled.min():.4f} - {input_scaled.max():.4f}")

# Predict
prediction_scaled = model.predict(input_scaled)[0]
print(f"\n🎯 PREDICTION:")
print(f"   Raw prediction: {prediction_scaled:.4f}")

# Vấn đề: Prediction đang trả về giá trị đã scale!
# Cần inverse transform hoặc train lại model với target scaling riêng

print(f"\n❌ VẤN ĐỀ:")
print(f"   Prediction: {prediction_scaled:.4f}")
print(f"   Actual: ${actual_target:,.2f}")
print(f"   Tỷ lệ: {prediction_scaled/actual_target:.6f}")

# Kiểm tra xem có target scaler không
print(f"\n🔍 KIỂM TRA TARGET SCALING:")
print("   Cần tạo target scaler riêng để inverse transform predictions")

# Tạo target scaler
from sklearn.preprocessing import StandardScaler
target_scaler = StandardScaler()

# Load lại dữ liệu training để fit target scaler
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

X, y = prepare_ml_data(df, lookback=10)
print(f"   Training targets range: {y.min():,.2f} - {y.max():,.2f}")

# Fit target scaler
target_scaler.fit(y.reshape(-1, 1))
print(f"   Target scaler mean: {target_scaler.mean_[0]:,.2f}")
print(f"   Target scaler scale: {target_scaler.scale_[0]:,.2f}")

# Test inverse transform
prediction_real = target_scaler.inverse_transform([[prediction_scaled]])[0][0]
print(f"\n✅ SAU KHI INVERSE TRANSFORM:")
print(f"   Prediction: ${prediction_real:,.2f}")
print(f"   Actual: ${actual_target:,.2f}")
print(f"   Error: ${abs(prediction_real - actual_target):,.2f}")
print(f"   Error %: {abs(prediction_real - actual_target)/actual_target*100:.2f}%")

# Lưu target scaler
with open("model_checkpoints/target_scaler.pkl", "wb") as f:
    pickle.dump(target_scaler, f)
print(f"\n💾 Đã lưu target_scaler.pkl") 