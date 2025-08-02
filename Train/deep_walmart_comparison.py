import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# Import các thuật toán ML để so sánh
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

checkpoint_dir = 'model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# 🔄 Đọc best R2 trước đó nếu có
r2_path = os.path.join(checkpoint_dir, "best_r2_score.txt")
if os.path.exists(r2_path):
    with open(r2_path, 'r') as f:
        best_r2_score = float(f.read().strip())
    print(f"📌 R² tốt nhất từ trước: {best_r2_score:.4f}")
else:
    best_r2_score = float('-inf')
    print("📌 Chưa có mô hình tốt nhất trước đó, sẽ bắt đầu ghi mới.")

# ========== 1. Dataset ==========
class WalmartDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

# ========== 2. GRU Model ==========
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):  # Tăng hidden_size và layers
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.linear(out)

# ========== 3. ML Models ==========
class MLModels:
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        self.scalers = {}
        self.trained_models = {}
    
    def prepare_ml_data(self, df, lookback=10):  # Thay đổi từ 12 thành 10
        """Chuẩn bị dữ liệu cho ML models (flatten sequences)"""
        feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
        
        all_features = []
        all_targets = []
        
        for store_id in df['Store'].unique():
            store_df = df[df['Store'] == store_id].sort_values('Week_Index')
            data = store_df[feature_cols].values.astype(np.float32)
            
            # Tạo sequences cho ML (flatten thành 1D)
            for i in range(len(data) - lookback):
                seq = data[i:i+lookback].flatten()  # Flatten thành 1D array
                target = data[i+lookback, 0]  # Weekly_Sales
                all_features.append(seq)
                all_targets.append(target)
        
        return np.array(all_features), np.array(all_targets)
    
    def train_ml_models(self, df, lookback=10):  # Thay đổi từ 12 thành 10
        """Train tất cả ML models"""
        print("\n===== Training ML Models =====")
        
        # Chuẩn bị dữ liệu
        X, y = self.prepare_ml_data(df, lookback)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features cho training
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['ml_scaler'] = scaler
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n--- Training {name} ---")
            
            # Train model trên dữ liệu scaled features, raw targets
            model.fit(X_train_scaled, y_train)  # y_train là raw values
            
            # Predict trên dữ liệu scaled features
            y_pred = model.predict(X_test_scaled)  # Predict raw values
            
            # Calculate metrics trên giá trị thực tế
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
        
        self.trained_models = results
        return results

# ========== 4. Prepare data ==========
def create_sequences(data, lookback):
    sequences, targets = [], []
    for i in range(len(data) - lookback):
        seq = data[i:i+lookback]
        target = data[i+lookback, 0]  # Weekly_Sales
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# ========== 5. Train GRU function ==========
def train_model_continual_gru(df, lookback=10):  # Thay đổi từ 12 thành 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
    
    input_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    df[['Weekly_Sales']] = target_scaler.fit_transform(df[['Weekly_Sales']])

    # Scale 7 features còn lại
    input_features = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
    input_scaler = MinMaxScaler()
    df[input_features] = input_scaler.fit_transform(df[input_features])

    store_ids = df['Store'].unique()
    model = GRUModel(input_size=len(feature_cols)).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Giảm learning rate
    scheduler = ExponentialLR(optimizer, gamma=0.98)  # Giảm gamma

    store_loss_log = {}

    for i, store_id in enumerate(store_ids):
        print(f"\n===== Training GRU on Store {store_id} =====")
        store_df = df[df['Store'] == store_id].sort_values('Week_Index')
        data = store_df[feature_cols].values.astype(np.float32)

        X, y = create_sequences(data, lookback)
        dataset = WalmartDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)  # Tăng batch_size, thêm shuffle

        # 🔒 Giai đoạn đầu: chỉ train Linear, giữ nguyên GRU
        if i <= 1:
            for name, param in model.named_parameters():
                if 'gru' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        # 🔓 Sau vài store: cho phép fine-tune GRU nhẹ nhàng
        else:
            for name, param in model.named_parameters():
                param.requires_grad = True  # fine-tune toàn bộ

        for epoch in range(20):  # Tăng epochs
            model.train()
            losses = []
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            print(f"[Epoch {epoch+1}] Loss: {mean_loss:.4f}")
        
        scheduler.step()
        store_loss_log[store_id] = mean_loss

    print("\n===== Final Store Losses =====")
    for store_id, loss in store_loss_log.items():
        print(f"Store {store_id}: Loss = {loss:.4f}")

    return model, input_scaler, target_scaler

# ========== 6. Evaluation GRU ==========
def evaluate_gru(df, trained_model, input_scaler, target_scaler, lookback=10, use_test_split=False):  # Thêm parameter
    print("\n===== Evaluating GRU Model =====")
    all_preds = []
    all_targets = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_test_split:
        # Sử dụng test split như ML
        print("🔍 GRU Evaluation: Using test split (like ML)")
        all_sequences = []
        all_targets_raw = []
        
        # Thu thập tất cả sequences từ tất cả stores
        for store in df['Store'].unique():
            feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
            df_store = pd.DataFrame(df[df['Store'] == store]).sort_values(by='Week_Index')
            data = df_store[feature_cols].values.astype(np.float32)
            
            X, y = create_sequences(data, lookback)
            all_sequences.extend(X)
            all_targets_raw.extend(y)
        
        # Split train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(all_sequences), np.array(all_targets_raw), 
            test_size=0.2, random_state=42
        )
        
        # Chỉ evaluate trên test set
        dataset = WalmartDataset(X_test, y_test)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = trained_model(inputs)
                all_preds.append(outputs.cpu().item())
                all_targets.append(targets.cpu().item())
    else:
        # Evaluation trên toàn bộ dữ liệu (như cũ)
        print("🔍 GRU Evaluation: Using all data (original)")
        with torch.no_grad():
            for store in df['Store'].unique():
                feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
                df_store = pd.DataFrame(df[df['Store'] == store]).sort_values(by='Week_Index')
                data = df_store[feature_cols].values.astype(np.float32)

                X, y = create_sequences(data, lookback)
                dataset = WalmartDataset(X, y)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

                for inputs, targets in dataloader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = trained_model(inputs)
                    all_preds.append(outputs.cpu().item())
                    all_targets.append(targets.cpu().item())

    # Convert to NumPy arrays
    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # Metrics
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    print(f"GRU Model Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    return {
        'model': 'GRU',
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': preds,
        'actuals': targets
    }

# ========== 7. Comparison Function ==========
def compare_models(gru_results, ml_results):
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    # Tạo DataFrame để so sánh
    comparison_data = []
    
    # Thêm GRU results
    comparison_data.append({
        'Model': 'GRU (Deep Learning)',
        'RMSE': gru_results['rmse'],
        'MAE': gru_results['mae'],
        'R²': gru_results['r2']
    })
    
    # Thêm ML results
    for name, results in ml_results.items():
        comparison_data.append({
            'Model': name,
            'RMSE': results['rmse'],
            'MAE': results['mae'],
            'R²': results['r2']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sắp xếp theo R² (tốt nhất lên đầu)
    comparison_df = comparison_df.sort_values('R²', ascending=False)
    
    print(comparison_df.to_string(index=False))
    
    # Tìm model tốt nhất
    best_model = comparison_df.iloc[0]
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   R² Score: {best_model['R²']:.4f}")
    print(f"   RMSE: {best_model['RMSE']:.4f}")
    print(f"   MAE: {best_model['MAE']:.4f}")
    
    return comparison_df

# ========== 7.5. Visualization Function ==========
def create_comparison_plots(gru_results, ml_results, df, lookback=10):
    """Tạo biểu đồ so sánh các mô hình với thực tế trên 20% test set"""
    print("\n" + "="*60)
    print("TẠO BIỂU ĐỒ SO SÁNH MÔ HÌNH")
    print("="*60)
    
    # Chuẩn bị dữ liệu test set
    feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
    
    # Thu thập tất cả sequences
    all_sequences = []
    all_targets_raw = []
    
    for store in df['Store'].unique():
        df_store = pd.DataFrame(df[df['Store'] == store]).sort_values(by='Week_Index')
        data = df_store[feature_cols].values.astype(np.float32)
        X, y = create_sequences(data, lookback)
        all_sequences.extend(X)
        all_targets_raw.extend(y)
    
    # Split train/test với 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(all_sequences), np.array(all_targets_raw), 
        test_size=0.2, random_state=42
    )
    
    print(f"✅ Test set size: {len(X_test)} samples")
    
    # Chuẩn bị predictions cho từng model
    predictions = {}
    actuals = y_test
    
    # 1. GRU Predictions
    print("🔍 Tính predictions cho GRU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gru_model = gru_results.get('model')  # Lấy model từ results
    
    if gru_model is not None:
        gru_preds = []
        test_dataset = WalmartDataset(X_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs = inputs.to(device)
                outputs = gru_model(inputs)
                gru_preds.append(outputs.cpu().item())
        
        predictions['GRU (Deep Learning)'] = np.array(gru_preds)
    
    # 2. ML Models Predictions
    print("🔍 Tính predictions cho ML models...")
    ml_models = MLModels()
    ml_models.train_ml_models(df, lookback)  # Train lại để lấy models
    
    # Scale test data cho ML
    scaler = ml_models.scalers['ml_scaler']
    X_test_scaled = scaler.transform(X_test)
    
    for name, results in ml_models.trained_models.items():
        model = results['model']
        y_pred = model.predict(X_test_scaled)
        predictions[name] = y_pred
    
    # 3. Tạo biểu đồ so sánh
    print("📊 Tạo biểu đồ so sánh...")
    
    # Biểu đồ 1: Line chart so sánh tất cả models
    plt.figure(figsize=(15, 10))
    
    # Plot actual values
    plt.subplot(2, 2, 1)
    plt.plot(actuals[:50], 'k-', linewidth=2, label='Actual Values', alpha=0.8)
    
    # Plot predictions cho từng model
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (model_name, preds) in enumerate(predictions.items()):
        plt.plot(preds[:50], color=colors[i], linewidth=1.5, label=f'{model_name}', alpha=0.7)
    
    plt.title('So sánh Predictions vs Actual Values (50 samples đầu)')
    plt.xlabel('Sample Index')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 2: Scatter plot Actual vs Predicted
    plt.subplot(2, 2, 2)
    for i, (model_name, preds) in enumerate(predictions.items()):
        plt.scatter(actuals, preds, alpha=0.6, s=20, label=f'{model_name}', color=colors[i])
    
    # Đường y=x (perfect prediction)
    min_val = min(actuals.min(), min([preds.min() for preds in predictions.values()]))
    max_val = max(actuals.max(), max([preds.max() for preds in predictions.values()]))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Weekly Sales')
    plt.ylabel('Predicted Weekly Sales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 3: Error distribution
    plt.subplot(2, 2, 3)
    for i, (model_name, preds) in enumerate(predictions.items()):
        errors = actuals - preds
        plt.hist(errors, bins=30, alpha=0.6, label=f'{model_name}', color=colors[i])
    
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 4: R² comparison
    plt.subplot(2, 2, 4)
    model_names = list(predictions.keys())
    r2_scores = []
    
    for model_name in model_names:
        preds = predictions[model_name]
        r2 = r2_score(actuals, preds)
        r2_scores.append(r2)
    
    bars = plt.bar(model_names, r2_scores, color=colors[:len(model_names)], alpha=0.7)
    plt.title('R² Score Comparison')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Thêm giá trị R² trên bars
    for bar, r2 in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{r2:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Tạo bảng metrics chi tiết
    print("\n📊 BẢNG METRICS CHI TIẾT:")
    print("-" * 80)
    print(f"{'Model':<25} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'MAPE':<10}")
    print("-" * 80)
    
    for model_name, preds in predictions.items():
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mae = mean_absolute_error(actuals, preds)
        r2 = r2_score(actuals, preds)
        mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
        
        print(f"{model_name:<25} {rmse:<10.4f} {mae:<10.4f} {r2:<10.4f} {mape:<10.2f}%")
    
    # 5. Tạo biểu đồ time series (nếu có đủ dữ liệu)
    if len(actuals) > 20:
        plt.figure(figsize=(15, 6))
        
        # Lấy 20 samples đầu để dễ nhìn
        sample_size = min(20, len(actuals))
        x_axis = range(sample_size)
        
        plt.plot(x_axis, actuals[:sample_size], 'ko-', linewidth=2, markersize=6, label='Actual', alpha=0.8)
        
        for i, (model_name, preds) in enumerate(predictions.items()):
            plt.plot(x_axis, preds[:sample_size], color=colors[i], linewidth=1.5, 
                    marker='o', markersize=4, label=f'{model_name}', alpha=0.7)
        
        plt.title('Time Series Comparison: Predictions vs Actual Values')
        plt.xlabel('Time Steps')
        plt.ylabel('Weekly Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('time_series_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"\n✅ Đã lưu biểu đồ so sánh vào:")
    print(f"   - model_comparison_plots.png")
    print(f"   - time_series_comparison.png")
    
    return predictions, actuals

# ========== 8. Prediction Function ==========
def predict_all_models(input_df, trained_model, input_scaler, target_scaler, ml_models, ml_results):
    """Dự đoán cho tất cả models sử dụng dữ liệu từ test.xlsx"""
    print("\n" + "="*60)
    print("PREDICTION COMPARISON USING TEST.XLSX")
    print("="*60)
    
    # Kiểm tra dữ liệu input
    if input_df.shape[0] != 10:
        raise ValueError("❌ File phải chứa đúng 10 tuần gần nhất!")
    if input_df.shape[1] != 8:
        raise ValueError("❌ File phải có đúng 8 cột: Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, WeekOfYear, Month")

    print("✅ Đã load dữ liệu thành công:\n", input_df)
    
    # Chuyển về numpy array
    input_weeks = input_df.to_numpy()
    
    # ========== GRU PREDICTION ==========
    print("\n--- GRU (Deep Learning) Prediction ---")
    feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
    input_features = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
    weekly_sales_col = ['Weekly_Sales']

    input_weeks_df = pd.DataFrame(input_weeks, columns=feature_cols)

    # Scale từng phần cho GRU
    scaled_inputs_only = input_scaler.transform(input_weeks_df[input_features])
    scaled_weekly_sales = target_scaler.transform(input_weeks_df[weekly_sales_col])

    # Ghép lại thành đúng input cho model: Weekly_Sales trước + các feature sau
    scaled_input = np.concatenate([scaled_weekly_sales, scaled_inputs_only], axis=1)

    # Dự đoán GRU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)
    trained_model.eval()
    with torch.no_grad():
        y_pred_scaled_gru = trained_model(input_tensor).detach().cpu().numpy()

    # Inverse transform để lấy giá trị thật sự
    y_pred_real_gru = target_scaler.inverse_transform(y_pred_scaled_gru)[0, 0]
    
    print(f"GRU Prediction: {y_pred_real_gru:.2f}")
    
    # ========== ML PREDICTIONS ==========
    print("\n--- ML Models Predictions ---")
    
    # Chuẩn bị dữ liệu cho ML (flatten 10 tuần)
    ml_input = input_weeks.flatten().reshape(1, -1)  # Shape: (1, 80) - 10 tuần x 8 features
    
    # Debug: In ra shape và giá trị
    print(f"🔍 Debug ML Input:")
    print(f"   Shape: {ml_input.shape}")
    print(f"   Sample values: {ml_input[0, :5]}...")
    
    # Scale cho ML
    ml_scaler = ml_models.scalers['ml_scaler']
    ml_input_scaled = ml_scaler.transform(ml_input)
    
    print(f"🔍 Debug ML Scaled Input:")
    print(f"   Shape: {ml_input_scaled.shape}")
    print(f"   Sample values: {ml_input_scaled[0, :5]}...")
    
    ml_predictions = {}
    
    for name, results in ml_results.items():
        model = results['model']
        # Predict trên dữ liệu scaled features
        y_pred = model.predict(ml_input_scaled)[0]  # Predict raw values directly
        
        print(f"🔍 Debug {name}:")
        print(f"   Raw prediction: {y_pred}")
        
        ml_predictions[name] = y_pred
        print(f"{name}: {y_pred:.2f}")
    
    # ========== COMPARISON TABLE ==========
    print("\n" + "="*60)
    print("PREDICTION COMPARISON TABLE")
    print("="*60)
    
    comparison_data = []
    
    # Thêm GRU prediction
    comparison_data.append({
        'Model': 'GRU (Deep Learning)',
        'Predicted Sales': f"{y_pred_real_gru:.2f}",
        'Method': 'Sequential Learning'
    })
    
    # Thêm ML predictions
    for name, pred in ml_predictions.items():
        comparison_data.append({
            'Model': name,
            'Predicted Sales': f"{pred:.2f}",
            'Method': 'Batch Learning'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Tìm prediction cao nhất và thấp nhất
    all_predictions = [y_pred_real_gru] + list(ml_predictions.values())
    max_pred = max(all_predictions)
    min_pred = min(all_predictions)
    
    print(f"\n📊 Prediction Range:")
    print(f"   Highest: {max_pred:.2f}")
    print(f"   Lowest: {min_pred:.2f}")
    print(f"   Difference: {max_pred - min_pred:.2f}")
    
    return {
        'gru_prediction': y_pred_real_gru,
        'ml_predictions': ml_predictions,
        'comparison_df': comparison_df
    }

# ========== 9. Main Execution ==========
if __name__ == "__main__":
    print("🚀 Starting Model Comparison: GRU vs Traditional ML")
    
    # Đọc dữ liệu
    df = pd.read_csv("walmart_processed_by_week.csv")
    
    # Train và evaluate GRU
    trained_model, input_scaler, target_scaler = train_model_continual_gru(df, lookback=10)
    
    # Chọn mode evaluation cho GRU
    use_fair_comparison = True  # Set True để so sánh công bằng với ML
    gru_results = evaluate_gru(df, trained_model, input_scaler, target_scaler, lookback=10, use_test_split=use_fair_comparison)
    
    # Train và evaluate ML models
    ml_models = MLModels()
    ml_results = ml_models.train_ml_models(df, lookback=10)
    
    # So sánh kết quả training
    comparison_df = compare_models(gru_results, ml_results)
    
    # ✅ THÊM VẼ BIỂU ĐỒ SO SÁNH
    print("\n" + "="*60)
    print("TẠO BIỂU ĐỒ SO SÁNH MÔ HÌNH")
    print("="*60)
    
    # Lưu model vào gru_results để vẽ biểu đồ
    gru_results['model'] = trained_model
    predictions, actuals = create_comparison_plots(gru_results, ml_results, df, lookback=10)
    
    # Lưu kết quả so sánh
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    print(f"\n✅ Comparison results saved to 'model_comparison_results.csv'")
    
    # ========== PREDICTION PHASE ==========
    print("\n📂 Đang đọc dữ liệu từ file 'test.xlsx'...")
    
    try:
        # Đọc file test.xlsx
        input_df = pd.read_excel("test.xlsx")
        
        # Thực hiện dự đoán cho tất cả models
        prediction_results = predict_all_models(
            input_df, trained_model, input_scaler, target_scaler, ml_models, ml_results
        )
        
        # Lưu kết quả dự đoán
        prediction_results['comparison_df'].to_csv('prediction_comparison_results.csv', index=False)
        print(f"\n✅ Prediction results saved to 'prediction_comparison_results.csv'")
        
    except FileNotFoundError:
        print("❌ Không tìm thấy file 'test.xlsx'. Bỏ qua phần dự đoán.")
    except Exception as e:
        print(f"❌ Lỗi khi đọc file test.xlsx: {e}")
    
    # Lưu model tốt nhất nếu là GRU
    if gru_results['r2'] > best_r2_score:
        best_r2_score = gru_results['r2']
        
        # ✅ Lưu mô hình GRU
        torch.save(trained_model.state_dict(), f"{checkpoint_dir}/best_gru_model.pth")
        
        # ✅ Lưu scaler
        with open(f"{checkpoint_dir}/input_scaler.pkl", 'wb') as f:
            pickle.dump(input_scaler, f)
        with open(f"{checkpoint_dir}/target_scaler.pkl", 'wb') as f:
            pickle.dump(target_scaler, f)

        # ✅ Ghi lại best R² vào file
        with open(r2_path, 'w') as f:
            f.write(str(best_r2_score))

        print(f"✅ Lưu GRU model mới với R² = {gru_results['r2']:.4f}!")
    
    # Lưu ML models tốt nhất
    best_ml_model_name = max(ml_results.keys(), key=lambda x: ml_results[x]['r2'])
    best_ml_model = ml_results[best_ml_model_name]['model']
    
    with open(f"{checkpoint_dir}/best_ml_model.pkl", 'wb') as f:
        pickle.dump(best_ml_model, f)
    
    with open(f"{checkpoint_dir}/ml_scaler.pkl", 'wb') as f:
        pickle.dump(ml_models.scalers['ml_scaler'], f)
    
    print(f"✅ Lưu ML model tốt nhất: {best_ml_model_name}")

# ========== 10. FastAPI (giữ nguyên logic) ==========
app = FastAPI()

class WeeklyInput(BaseModel):
    data: List[List[float]]  # 10 hàng x 8 cột

@app.post("/predict")
def predict_sales(input_data: WeeklyInput):
    try:
        if len(input_data.data) != 10:
            raise HTTPException(status_code=400, detail="❌ Cần đúng 10 tuần dữ liệu (10 dòng).")
        
        feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
                        'CPI', 'Unemployment', 'WeekOfYear', 'Month']
        input_df = pd.DataFrame(input_data.data, columns=feature_cols)
        input_features = feature_cols[1:]
        weekly_sales_col = ['Weekly_Sales']

        # Scale dữ liệu
        scaled_inputs_only = input_scaler.transform(input_df[input_features])
        scaled_weekly_sales = target_scaler.transform(input_df[weekly_sales_col])
        scaled_input = np.concatenate([scaled_weekly_sales, scaled_inputs_only], axis=1)

        # Dự đoán
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)
        trained_model.eval()
        with torch.no_grad():
            y_pred_scaled = trained_model(input_tensor).detach().cpu().numpy()
        y_pred_real = target_scaler.inverse_transform(y_pred_scaled)[0, 0]

        return {"predicted_weekly_sales": round(float(y_pred_real), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("deep_walmart_comparison:app", host="0.0.0.0", port=8000, reload=True) 