import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pickle
import os

# Tạo thư mục output
os.makedirs('test_visualization', exist_ok=True)

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

def train_and_test_models(df, lookback=10):
    """Train models và test trên 20% dữ liệu"""
    print("="*60)
    print("TRAINING VÀ TESTING MODELS TRÊN 20% DỮ LIỆU")
    print("="*60)
    
    # Chuẩn bị dữ liệu
    X, y = prepare_ml_data(df, lookback)
    print(f"📊 Tổng số samples: {len(X)}")
    print(f"📊 Features per sample: {X.shape[1]}")
    
    # Chia train/test với 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Train set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Định nghĩa các models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    # Train và test từng model
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict trên test set
        y_pred = model.predict(X_test_scaled)
        
        # Tính metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred,
            'actuals': y_test
        }
        
        predictions[name] = y_pred
        
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  MAE:  ${mae:,.2f}")
        print(f"  R²:   {r2:.4f}")
    
    return results, predictions, y_test, scaler

def create_test_visualizations(results, predictions, y_test):
    """Tạo biểu đồ so sánh models với test data"""
    print("\n" + "="*60)
    print("TẠO BIỂU ĐỒ SO SÁNH VỚI TEST DATA")
    print("="*60)
    
    # 1. Biểu đồ so sánh predictions vs actual (line chart)
    plt.figure(figsize=(20, 15))
    
    # Lấy 50 samples đầu để dễ nhìn
    sample_size = min(50, len(y_test))
    x_axis = range(sample_size)
    
    # Plot actual values
    plt.subplot(3, 2, 1)
    plt.plot(x_axis, y_test[:sample_size], 'ko-', linewidth=3, markersize=8, 
             label='Actual Test Values', alpha=0.8)
    
    # Plot predictions cho từng model
    colors = ['red', 'blue', 'green', 'orange']
    for i, (model_name, preds) in enumerate(predictions.items()):
        plt.plot(x_axis, preds[:sample_size], color=colors[i], linewidth=2, 
                marker='o', markersize=4, label=f'{model_name}', alpha=0.7)
    
    plt.title('So sánh Predictions vs Actual Test Values (50 samples đầu)', fontsize=14)
    plt.xlabel('Test Sample Index')
    plt.ylabel('Weekly Sales ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Scatter plot Actual vs Predicted
    plt.subplot(3, 2, 2)
    for i, (model_name, preds) in enumerate(predictions.items()):
        plt.scatter(y_test, preds, alpha=0.6, s=30, label=f'{model_name}', color=colors[i])
    
    # Đường y=x (perfect prediction)
    min_val = min(y_test.min(), min([preds.min() for preds in predictions.values()]))
    max_val = max(y_test.max(), max([preds.max() for preds in predictions.values()]))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Perfect Prediction')
    
    plt.title('Actual vs Predicted Values (Test Set)', fontsize=14)
    plt.xlabel('Actual Weekly Sales ($)')
    plt.ylabel('Predicted Weekly Sales ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Error distribution
    plt.subplot(3, 2, 3)
    for i, (model_name, preds) in enumerate(predictions.items()):
        errors = y_test - preds
        plt.hist(errors, bins=30, alpha=0.6, label=f'{model_name}', color=colors[i])
    
    plt.title('Error Distribution (Test Set)', fontsize=14)
    plt.xlabel('Prediction Error ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. R² comparison
    plt.subplot(3, 2, 4)
    model_names = list(predictions.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    
    bars = plt.bar(model_names, r2_scores, color=colors[:len(model_names)], alpha=0.7)
    plt.title('R² Score Comparison (Test Set)', fontsize=14)
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Thêm giá trị R² trên bars
    for bar, r2 in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. RMSE comparison
    plt.subplot(3, 2, 5)
    rmse_scores = [results[name]['rmse'] for name in model_names]
    
    bars = plt.bar(model_names, rmse_scores, color=colors[:len(model_names)], alpha=0.7)
    plt.title('RMSE Comparison (Test Set)', fontsize=14)
    plt.ylabel('RMSE ($)')
    plt.xticks(rotation=45)
    
    # Thêm giá trị RMSE trên bars
    for bar, rmse in zip(bars, rmse_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                f'${rmse:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Time series comparison (20 samples)
    plt.subplot(3, 2, 6)
    sample_size = min(20, len(y_test))
    x_axis = range(sample_size)
    
    plt.plot(x_axis, y_test[:sample_size], 'ko-', linewidth=3, markersize=8, 
             label='Actual Test Values', alpha=0.8)
    
    for i, (model_name, preds) in enumerate(predictions.items()):
        plt.plot(x_axis, preds[:sample_size], color=colors[i], linewidth=2, 
                marker='s', markersize=4, label=f'{model_name}', alpha=0.7)
    
    plt.title('Time Series Comparison (20 samples)', fontsize=14)
    plt.xlabel('Time Steps')
    plt.ylabel('Weekly Sales ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_visualization/model_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Tạo biểu đồ chi tiết cho từng model
    create_individual_model_plots(results, predictions, y_test)
    
    # 8. Tạo bảng metrics chi tiết
    create_metrics_table(results)
    
    return results

def create_individual_model_plots(results, predictions, y_test):
    """Tạo biểu đồ chi tiết cho từng model"""
    print("\n📊 Tạo biểu đồ chi tiết cho từng model...")
    
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, (model_name, preds) in enumerate(predictions.items()):
        plt.figure(figsize=(15, 10))
        
        # 1. Line chart comparison
        plt.subplot(2, 3, 1)
        sample_size = min(30, len(y_test))
        x_axis = range(sample_size)
        
        plt.plot(x_axis, y_test[:sample_size], 'ko-', linewidth=3, markersize=8, 
                label='Actual Test Values', alpha=0.8)
        plt.plot(x_axis, preds[:sample_size], color=colors[i], linewidth=2, 
                marker='o', markersize=4, label=f'{model_name} Predictions', alpha=0.7)
        
        plt.title(f'{model_name} vs Actual (30 samples)', fontsize=14)
        plt.xlabel('Test Sample Index')
        plt.ylabel('Weekly Sales ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Scatter plot
        plt.subplot(2, 3, 2)
        plt.scatter(y_test, preds, alpha=0.6, s=30, color=colors[i])
        
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2)
        
        plt.title(f'{model_name}: Actual vs Predicted', fontsize=14)
        plt.xlabel('Actual Weekly Sales ($)')
        plt.ylabel('Predicted Weekly Sales ($)')
        plt.grid(True, alpha=0.3)
        
        # 3. Error distribution
        plt.subplot(2, 3, 3)
        errors = y_test - preds
        plt.hist(errors, bins=30, alpha=0.7, color=colors[i])
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.title(f'{model_name}: Error Distribution', fontsize=14)
        plt.xlabel('Prediction Error ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 4. Residual plot
        plt.subplot(2, 3, 4)
        plt.scatter(preds, errors, alpha=0.6, s=30, color=colors[i])
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.title(f'{model_name}: Residual Plot', fontsize=14)
        plt.xlabel('Predicted Values ($)')
        plt.ylabel('Residuals ($)')
        plt.grid(True, alpha=0.3)
        
        # 5. Error vs Actual
        plt.subplot(2, 3, 5)
        plt.scatter(y_test, errors, alpha=0.6, s=30, color=colors[i])
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.title(f'{model_name}: Error vs Actual', fontsize=14)
        plt.xlabel('Actual Values ($)')
        plt.ylabel('Prediction Error ($)')
        plt.grid(True, alpha=0.3)
        
        # 6. Metrics summary
        plt.subplot(2, 3, 6)
        metrics = results[model_name]
        
        # Tạo text box với metrics
        metrics_text = f"""
        R² Score: {metrics['r2']:.4f}
        RMSE: ${metrics['rmse']:,.2f}
        MAE: ${metrics['mae']:,.2f}
        
        Test Samples: {len(y_test)}
        Mean Actual: ${y_test.mean():,.2f}
        Mean Predicted: ${preds.mean():,.2f}
        """
        
        plt.text(0.1, 0.5, metrics_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.title(f'{model_name}: Metrics Summary', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'test_visualization/{model_name.replace(" ", "_").lower()}_detailed.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()

def create_metrics_table(results):
    """Tạo bảng metrics chi tiết"""
    print("\n" + "="*80)
    print("BẢNG METRICS CHI TIẾT (TEST SET)")
    print("="*80)
    print(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'R²':<10} {'MAPE':<10}")
    print("-" * 80)
    
    for name, res in results.items():
        rmse = res['rmse']
        mae = res['mae']
        r2 = res['r2']
        mape = np.mean(np.abs((res['actuals'] - res['predictions']) / res['actuals'])) * 100
        
        print(f"{name:<20} ${rmse:<11,.2f} ${mae:<11,.2f} {r2:<10.4f} {mape:<10.2f}%")
    
    # Tìm model tốt nhất
    best_model = max(results.items(), key=lambda x: x[1]['r2'])
    print("-" * 80)
    print(f"🏆 BEST MODEL: {best_model[0]}")
    print(f"   R² Score: {best_model[1]['r2']:.4f}")
    print(f"   RMSE: ${best_model[1]['rmse']:,.2f}")
    print(f"   MAE: ${best_model[1]['mae']:,.2f}")
    
    # Lưu kết quả vào CSV
    metrics_data = []
    for name, res in results.items():
        mape = np.mean(np.abs((res['actuals'] - res['predictions']) / res['actuals'])) * 100
        metrics_data.append({
            'Model': name,
            'RMSE': res['rmse'],
            'MAE': res['mae'],
            'R²': res['r2'],
            'MAPE': mape
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.sort_values('R²', ascending=False)
    metrics_df.to_csv('test_visualization/test_metrics_results.csv', index=False)
    print(f"\n✅ Đã lưu metrics vào 'test_visualization/test_metrics_results.csv'")

def main():
    """Hàm main để chạy toàn bộ quá trình testing và visualization"""
    print("🚀 BẮT ĐẦU TESTING VÀ VISUALIZATION")
    print("="*60)
    
    # Đọc dữ liệu
    df = pd.read_csv("walmart_processed_by_week.csv")
    print(f"✅ Đã load dữ liệu: {len(df)} records")
    
    # Train và test models
    results, predictions, y_test, scaler = train_and_test_models(df, lookback=10)
    
    # Tạo visualizations
    results = create_test_visualizations(results, predictions, y_test)
    
    # Lưu model tốt nhất
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]
    
    print(f"\n🎯 KẾT LUẬN:")
    print(f"   - Model tốt nhất: {best_model_name}")
    print(f"   - R² Score: {best_model['r2']:.4f}")
    print(f"   - RMSE: ${best_model['rmse']:,.2f}")
    print(f"   - MAE: ${best_model['mae']:,.2f}")
    
    print(f"\n✅ Đã lưu tất cả biểu đồ vào thư mục 'test_visualization/'")
    print(f"✅ Files được tạo:")
    print(f"   - model_test_comparison.png")
    print(f"   - *_detailed.png (cho từng model)")
    print(f"   - test_metrics_results.csv")

if __name__ == "__main__":
    main() 