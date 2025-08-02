import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pickle
import os

# Tạo thư mục lưu kết quả
os.makedirs('report_outputs', exist_ok=True)

# Đọc dữ liệu
df = pd.read_csv("walmart_processed_by_week.csv")

print("="*60)
print("BÁO CÁO HUẤN LUYỆN MÔ HÌNH AI")
print("="*60)

# ========== 1. CHUẨN BỊ DỮ LIỆU CHO ML ==========
print("\n1. CHUẨN BỊ DỮ LIỆU CHO MACHINE LEARNING")
print("-" * 50)

def prepare_ml_data(df, lookback=10):
    """Chuẩn bị dữ liệu cho ML với lookback window"""
    feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
    all_features = []
    all_targets = []
    
    for store_id in df['Store'].unique():
        store_df = df[df['Store'] == store_id].sort_values('Week_Index')
        data = store_df[feature_cols].values.astype(np.float32)
        
        for i in range(len(data) - lookback):
            seq = data[i:i+lookback].flatten()  # Features: 10 tuần x 8 features = 80 features
            target = data[i+lookback, 0]        # Target: Weekly_Sales của tuần tiếp theo
            all_features.append(seq)
            all_targets.append(target)
    
    return np.array(all_features), np.array(all_targets)

# Chuẩn bị dữ liệu
X, y = prepare_ml_data(df, lookback=10)
print(f"Kích thước dữ liệu:")
print(f"  - Features (X): {X.shape}")
print(f"  - Targets (y): {y.shape}")
print(f"  - Lookback window: 10 tuần")
print(f"  - Features per sample: {X.shape[1]} (10 tuần × 8 features)")

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  - Train set: {X_train.shape[0]} samples")
print(f"  - Test set: {X_test.shape[0]} samples")

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  - Đã chuẩn hóa dữ liệu bằng StandardScaler")

# ========== 2. CHỌN MÔ HÌNH ==========
print("\n2. CHỌN MÔ HÌNH AI")
print("-" * 50)

models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'description': 'Mô hình hồi quy tuyến tính cơ bản, dễ giải thích',
        'pros': 'Đơn giản, nhanh, interpretable',
        'cons': 'Không bắt được patterns phi tuyến'
    },
    'Random Forest': {
        'model': RandomForestRegressor(n_estimators=100, random_state=42),
        'description': 'Ensemble method với nhiều decision trees',
        'pros': 'Xử lý tốt outliers, feature importance',
        'cons': 'Có thể overfit với dữ liệu nhỏ'
    },
    'XGBoost': {
        'model': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'description': 'Gradient boosting với regularization',
        'pros': 'Hiệu suất cao, xử lý tốt dữ liệu lớn',
        'cons': 'Có thể overfit, cần tuning parameters'
    },
    'MLP (Neural Network)': {
        'model': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'description': 'Neural network với 2 hidden layers',
        'pros': 'Bắt được patterns phức tạp',
        'cons': 'Black box, cần nhiều dữ liệu'
    }
}

print("Các mô hình được thử nghiệm:")
for i, (name, info) in enumerate(models.items(), 1):
    print(f"{i}. {name}")
    print(f"   - Mô tả: {info['description']}")
    print(f"   - Ưu điểm: {info['pros']}")
    print(f"   - Nhược điểm: {info['cons']}")
    print()

# ========== 3. HUẤN LUYỆN MÔ HÌNH ==========
print("3. HUẤN LUYỆN MÔ HÌNH")
print("-" * 50)

results = {}
for name, info in models.items():
    print(f"\n--- Training {name} ---")
    model = info['model']
    
    # Training
    model.fit(X_train_scaled, y_train)
    
    # Prediction
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred,
        'actuals': y_test
    }
    
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  R²:   {r2:.4f}")

# ========== 4. SO SÁNH KẾT QUẢ ==========
print("\n4. SO SÁNH KẾT QUẢ CÁC MÔ HÌNH")
print("-" * 50)

# Tạo bảng so sánh
comparison_data = []
for name, res in results.items():
    comparison_data.append({
        'Model': name,
        'RMSE': res['rmse'],
        'MAE': res['mae'],
        'R²': res['r2']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('R²', ascending=False)

print("Bảng so sánh hiệu suất:")
print(comparison_df.to_string(index=False))

# Lưu kết quả
comparison_df.to_csv('report_outputs/model_comparison_results.csv', index=False)
print("\n✅ Đã lưu kết quả so sánh vào 'report_outputs/model_comparison_results.csv'")

# ========== 5. PHÂN TÍCH FEATURE IMPORTANCE ==========
print("\n5. PHÂN TÍCH FEATURE IMPORTANCE")
print("-" * 50)

# Tạo tên features
feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
lookback = 10
feature_names = []
for i in range(lookback):
    for col in feature_cols:
        feature_names.append(f"{col}_t-{lookback-i}")

# Vẽ feature importance cho từng model
for name, res in results.items():
    model = res['model']
    print(f"\n--- Feature Importance cho {name} ---")
    
    if name == "Linear Regression":
        importances = np.abs(model.coef_)
        title = f"Feature Coefficient Magnitude - {name}"
    elif name in ["Random Forest", "XGBoost"]:
        importances = model.feature_importances_
        title = f"Feature Importance - {name}"
    else:
        # Permutation importance cho MLP
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=1)
        importances = result.importances_mean
        title = f"Permutation Importance - {name}"
    
    # Vẽ top 15 features
    idx = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(idx)), importances[idx], color='skyblue')
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'report_outputs/feature_importance_{name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Top 5 features quan trọng nhất:")
    for i, idx_val in enumerate(idx[:5]):
        print(f"  {i+1}. {feature_names[idx_val]}: {importances[idx_val]:.4f}")

# ========== 6. LƯU MÔ HÌNH TỐT NHẤT ==========
print("\n6. LƯU MÔ HÌNH TỐT NHẤT")
print("-" * 50)

best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']

# Tạo thư mục model checkpoints
os.makedirs('model_checkpoints', exist_ok=True)

# Lưu model và scaler
with open("model_checkpoints/best_model.pkl", 'wb') as f:
    pickle.dump(best_model, f)
with open("model_checkpoints/scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Mô hình tốt nhất: {best_model_name}")
print(f"   - R² Score: {comparison_df.iloc[0]['R²']:.4f}")
print(f"   - RMSE: ${comparison_df.iloc[0]['RMSE']:,.2f}")
print(f"   - MAE: ${comparison_df.iloc[0]['MAE']:,.2f}")
print("✅ Đã lưu model và scaler vào 'model_checkpoints/'")

# ========== 7. PHÂN TÍCH LỖI ==========
print("\n7. PHÂN TÍCH LỖI")
print("-" * 50)

best_predictions = results[best_model_name]['predictions']
best_actuals = results[best_model_name]['actuals']

# Tính error
errors = best_actuals - best_predictions
error_percentage = np.abs(errors) / best_actuals * 100

print(f"Thống kê lỗi của {best_model_name}:")
print(f"  - Lỗi trung bình: ${np.mean(np.abs(errors)):,.2f}")
print(f"  - Lỗi tương đối trung bình: {np.mean(error_percentage):.1f}%")
print(f"  - Lỗi tương đối tối đa: {np.max(error_percentage):.1f}%")

# Vẽ biểu đồ lỗi
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(best_actuals, best_predictions, alpha=0.5)
plt.plot([best_actuals.min(), best_actuals.max()], [best_actuals.min(), best_actuals.max()], 'r--', lw=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted')

plt.subplot(1, 3, 2)
plt.hist(errors, bins=50, alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')

plt.subplot(1, 3, 3)
plt.hist(error_percentage, bins=50, alpha=0.7)
plt.xlabel('Relative Error (%)')
plt.ylabel('Frequency')
plt.title('Relative Error Distribution')

plt.tight_layout()
plt.savefig('report_outputs/error_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("HOÀN THÀNH BÁO CÁO MÔ HÌNH AI")
print("="*60) 