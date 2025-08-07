import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

checkpoint_dir = 'model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# ========== 1. Chuẩn bị dữ liệu ==========
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

# ========== 2. Train & Evaluate ML Models ==========
def train_and_evaluate_ml_models(df, lookback=10):
    print("\n===== Training ML Models =====")
    X, y_raw = prepare_ml_data(df, lookback)

    # Log transform target (KHONG scale)
    y_log = np.log1p(y_raw).reshape(-1)

    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }
    results = {}
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_scaled, y_train_log)
        y_pred_log = model.predict(X_test_scaled)

        # Inverse log1p
        y_pred_real = np.expm1(y_pred_log)
        y_test_real = np.expm1(y_test_log)

        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        mae = mean_absolute_error(y_test_real, y_pred_real)
        r2 = r2_score(y_test_real, y_pred_real)

        results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred_real,
            'actuals': y_test_real
        }
        print(f"{name}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
    return results, scaler

# ========== 3. Main ==========
if __name__ == "__main__":
    df = pd.read_csv("walmart_processed_by_week.csv")
    print("\n===== MÔ TẢ Dữ LIỆU BAN ĐẦU =====")
    print(df.describe(include='all'))

    results, scaler = train_and_evaluate_ml_models(df, lookback=10)

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
    print("\nMODEL COMPARISON RESULTS:")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv('ml_model_comparison_results.csv', index=False)
    print("\n✅ ML model comparison results saved to 'ml_model_comparison_results.csv'")

    best_model_name = comparison_df.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    with open(f"{checkpoint_dir}/best_ml_model.pkl", 'wb') as f:
        pickle.dump(best_model, f)
    with open(f"{checkpoint_dir}/ml_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    print(f"✅ Lưu ML model tốt nhất: {best_model_name}")
    print(f"✅ KHÔNG dùng target scaler nên không cần lưu")
import matplotlib.pyplot as plt

# === Vẽ dự đoán vs thực tế cho mô hình XGBoost ===
xgb_result = results['XGBoost']
y_pred = xgb_result['predictions']
y_true = xgb_result['actuals']

plt.figure(figsize=(12, 5))
plt.plot(y_true[:len(y_true)//5], label='Actual', marker='o')
plt.plot(y_pred[:len(y_pred)//5], label='Predicted', marker='x')
plt.title('🔍 So sánh XGBoost: Thực tế vs Dự đoán (20% đầu)')
plt.xlabel('Sample Index')
plt.ylabel('Weekly Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('xgboost_prediction_vs_actual.png')
plt.show()
