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
    X, y = prepare_ml_data(df, lookback)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
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
        print(f"{name}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
    return results, scaler

# ========== 3. Main ==========
if __name__ == "__main__":
    df = pd.read_csv("walmart_processed_by_week.csv")
        # ========== A. Thống kê mô tả dữ liệu ==========
    print("\n===== MÔ TẢ DỮ LIỆU BAN ĐẦU =====")
    print(df.describe(include='all'))
    
    # Vẽ biểu đồ phân tích dữ liệu
    print("\n===== VẼ BIỂU ĐỒ PHÂN TÍCH DỮ LIỆU =====")
    
    # 1. Biểu đồ phân phối Weekly_Sales
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(df['Weekly_Sales'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Phân phối Weekly Sales')
    plt.xlabel('Weekly Sales')
    plt.ylabel('Tần suất')
    
    # 2. Biểu đồ box plot Weekly_Sales theo Store
    plt.subplot(2, 3, 2)
    store_sales = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
    plt.bar(range(len(store_sales)), store_sales.values, color='lightcoral')
    plt.title('Doanh số trung bình theo Store')
    plt.xlabel('Store ID')
    plt.ylabel('Weekly Sales trung bình')
    plt.xticks(range(0, len(store_sales), 5))
    
    # 3. Biểu đồ Weekly_Sales theo thời gian
    plt.subplot(2, 3, 3)
    time_sales = df.groupby('Week_Index')['Weekly_Sales'].mean()
    plt.plot(time_sales.index, time_sales.values, color='green', linewidth=2)
    plt.title('Xu hướng Weekly Sales theo thời gian')
    plt.xlabel('Week Index')
    plt.ylabel('Weekly Sales trung bình')
    
    # 4. Biểu đồ Weekly_Sales theo Month
    plt.subplot(2, 3, 4)
    month_sales = df.groupby('Month')['Weekly_Sales'].mean()
    plt.bar(month_sales.index, month_sales.values, color='gold')
    plt.title('Doanh số trung bình theo tháng')
    plt.xlabel('Tháng')
    plt.ylabel('Weekly Sales trung bình')
    
    # 5. Biểu đồ Weekly_Sales theo Holiday_Flag
    plt.subplot(2, 3, 5)
    holiday_sales = df.groupby('Holiday_Flag')['Weekly_Sales'].mean()
    plt.bar(['Không phải ngày lễ', 'Ngày lễ'], holiday_sales.values, color=['lightblue', 'orange'])
    plt.title('Doanh số trung bình theo ngày lễ')
    plt.ylabel('Weekly Sales trung bình')
    
    # 6. Biểu đồ correlation matrix
    plt.subplot(2, 3, 6)
    numeric_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    correlation_matrix = df[numeric_cols].corr()
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.title('Ma trận tương quan')
    
    # Thêm giá trị correlation vào ô
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data_analysis_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Đã vẽ và lưu biểu đồ phân tích dữ liệu vào 'data_analysis_visualization.png'")

    # ========== B. Biểu đồ phân phối ==========
    import seaborn as sns

    numeric_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Histogram - {col}")
        plt.tight_layout()
        plt.savefig(f"histogram_{col}.png")
        plt.close()

    print("✅ Đã vẽ và lưu biểu đồ histogram cho các đặc trưng số.")

    # ========== C. Boxplot (phát hiện ngoại lệ) ==========
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x=col)
        plt.title(f"Boxplot - {col}")
        plt.tight_layout()
        plt.savefig(f"boxplot_{col}.png")
        plt.close()

    print("✅ Đã vẽ và lưu boxplot cho các đặc trưng số.")

    # ========== D. Line chart doanh số theo thời gian ==========
    plt.figure(figsize=(12, 5))
    df_grouped = df.groupby('Week_Index')['Weekly_Sales'].sum()
    df_grouped.plot()
    plt.title("Tổng Doanh Số Theo Tuần")
    plt.xlabel("Tuần")
    plt.ylabel("Doanh số")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("linechart_weekly_sales.png")
    plt.close()
    print("✅ Đã vẽ biểu đồ line chart tổng doanh số theo thời gian.")

    # ========== E. Ma trận tương quan ==========
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.close()
    print("✅ Đã vẽ và lưu heatmap ma trận tương quan giữa các đặc trưng.")

    results, scaler = train_and_evaluate_ml_models(df, lookback=10)
    # Lưu kết quả so sánh
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
    # Lưu model tốt nhất
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    with open(f"{checkpoint_dir}/best_ml_model.pkl", 'wb') as f:
        pickle.dump(best_model, f)
    with open(f"{checkpoint_dir}/ml_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # ✅ THÊM: Lưu target scaler
    from sklearn.preprocessing import StandardScaler
    target_scaler = StandardScaler()
    target_scaler.fit(y.reshape(-1, 1))
    with open(f"{checkpoint_dir}/target_scaler.pkl", 'wb') as f:
        pickle.dump(target_scaler, f)
    
    print(f"✅ Lưu ML model tốt nhất: {best_model_name}")
    print(f"✅ Lưu target scaler cho inverse transform")

    # ========== 4. Feature Importance Visualization ==========
    importances_dict = {}
    feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
    lookback = 10
    feature_names = []
    for i in range(lookback):
        for col in feature_cols:
            feature_names.append(f"{col}_t-{lookback-i}")

    X, y = prepare_ml_data(df, lookback)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, res in results.items():
        model = res['model']
        print(f"\n=== Feature Importance for {name} ===")
        if name == "Random Forest":
            importances = model.feature_importances_
            importances_dict[name] = importances
            idx = np.argsort(importances)[::-1][:15]
            plt.figure(figsize=(10,5))
            plt.title(f"Feature Importance - {name}")
            plt.bar(np.array(feature_names)[idx], importances[idx])
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f'feature_importance_{name.replace(" ", "_").lower()}.png')
            plt.show()
        elif name == "Linear Regression":
            importances = np.abs(model.coef_)
            importances_dict[name] = importances
            idx = np.argsort(importances)[::-1][:15]
            plt.figure(figsize=(10,5))
            plt.title(f"Feature Coefficient Magnitude - {name}")
            plt.bar(np.array(feature_names)[idx], importances[idx])
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f'feature_importance_{name.replace(" ", "_").lower()}.png')
            plt.show()
        elif name == "XGBoost":
            importances = model.feature_importances_
            importances_dict[name] = importances
            idx = np.argsort(importances)[::-1][:15]
            plt.figure(figsize=(10,5))
            plt.title(f"Feature Importance - {name}")
            plt.bar(np.array(feature_names)[idx], importances[idx])
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f'feature_importance_{name.replace(" ", "_").lower()}.png')
            plt.show()
        else:
            # Permutation importance cho SVR, MLP
            result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=1)
            importances = result.importances_mean
            importances_dict[name] = importances
            idx = np.argsort(importances)[::-1][:15]
            plt.figure(figsize=(10,5))
            plt.title(f"Permutation Importance - {name}")
            plt.bar(np.array(feature_names)[idx], importances[idx])
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f'feature_importance_{name.replace(" ", "_").lower()}.png')
            plt.show()
    print("\n✅ Đã lưu các biểu đồ feature importance cho từng model ML!")

    