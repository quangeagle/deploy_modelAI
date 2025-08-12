# GRU Sales Prediction Model
# Dự đoán doanh thu dựa trên dữ liệu quá khứ (các tuần trước)
# Tỷ lệ train-test: 80-20

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Cài đặt device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Sử dụng device: {device}")

# ========== 1. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU ==========
print("\n📊 Đang đọc dữ liệu Walmart...")
df = pd.read_csv("E:\TrainAI\Train\walmart_processed_by_week.csv")
df['Date'] = pd.to_datetime(df['Date'])

print(f"📈 Tổng số bản ghi: {len(df):,}")
print(f"🏪 Số lượng cửa hàng: {df['Store'].nunique()}")
print(f"📅 Thời gian: {df['Date'].min().strftime('%Y-%m-%d')} đến {df['Date'].max().strftime('%Y-%m-%d')}")

# ========== 2. TẠO SEQUENTIAL DATASET ==========
class WalmartSequentialDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

def create_sequential_data(df, lookback=10):
    """
    Tạo dữ liệu sequential cho GRU - chỉ sử dụng doanh thu các tuần trước
    - lookback: số tuần trước để dự đoán
    """
    print(f"\n🔄 Tạo sequential data với lookback={lookback} tuần...")
    print(f"📊 Chỉ sử dụng Weekly_Sales để dự đoán")
    
    all_sequences = []
    all_targets = []
    all_target_dates = []  # ngày của từng target (tuần được dự đoán)
    
    for store_id in df['Store'].unique():
        store_df = df[df['Store'] == store_id].sort_values('Date')
        
        # Chỉ lấy dữ liệu doanh thu
        sales_data = store_df['Weekly_Sales'].values.astype(np.float32)
        
        # Tạo sequences
        for i in range(len(sales_data) - lookback):
            sequence = sales_data[i:i+lookback]  # 10 tuần doanh thu trước
            target = sales_data[i+lookback]      # Doanh thu tuần tiếp theo
            target_date = store_df['Date'].iloc[i+lookback]
            
            # Reshape sequence để có shape (lookback, 1) thay vì (lookback,)
            sequence = sequence.reshape(-1, 1)
            
            all_sequences.append(sequence)
            all_targets.append(target)
            all_target_dates.append(target_date)
    
    return np.array(all_sequences), np.array(all_targets), pd.to_datetime(all_target_dates)

# ========== 3. GRU MODEL ==========
class GRUSalesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(GRUSalesPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.linear = nn.Linear(hidden_size, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        gru_out, _ = self.gru(x)
        
        # Lấy output của timestep cuối cùng
        last_output = gru_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Linear layer để dự đoán
        output = self.linear(last_output)
        
        return output

# ========== 4. TRAINING FUNCTION ==========
def train_gru_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """
    Huấn luyện GRU model
    """
    print(f"\n🚀 Bắt đầu training GRU model...")
    print(f"📊 Số epochs: {num_epochs}")
    print(f"📚 Learning rate: {learning_rate}")
    
    # Loss function và optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Lưu training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        # Tính average loss
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Lưu best model
            torch.save(model.state_dict(), 'best_gru_model.pth')
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if patience_counter >= 20:
            print(f"⏹️ Early stopping tại epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_gru_model.pth'))
    
    return model, train_losses, val_losses

# ========== 5. EVALUATION FUNCTION ==========
def evaluate_model(model, test_loader, scaler):
    """
    Đánh giá model trên test set
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(targets.numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Inverse transform nếu cần
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    return predictions, actuals, rmse, mae, r2

# ========== 6. VISUALIZATION FUNCTION ==========
def plot_training_history(train_losses, val_losses):
    """
    Vẽ biểu đồ training history
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training History (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gru_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions(predictions, actuals, sample_size=100):
    """
    Vẽ biểu đồ so sánh predictions vs actuals
    """
    # Lấy sample để vẽ
    if len(predictions) > sample_size:
        indices = np.random.choice(len(predictions), sample_size, replace=False)
        pred_sample = predictions[indices]
        actual_sample = actuals[indices]
    else:
        pred_sample = predictions
        actual_sample = actuals
    
    plt.figure(figsize=(15, 5))
    
    # Line plot
    plt.subplot(1, 2, 1)
    plt.plot(actual_sample, label='Actual', marker='o', alpha=0.7)
    plt.plot(pred_sample, label='Predicted', marker='x', alpha=0.7)
    plt.title('GRU Predictions vs Actual (Sample)')
    plt.xlabel('Sample Index')
    plt.ylabel('Weekly Sales ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(actual_sample, pred_sample, alpha=0.6)
    plt.plot([actual_sample.min(), actual_sample.max()], 
             [actual_sample.min(), actual_sample.max()], 'r--', label='Perfect Prediction')
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual Sales ($)')
    plt.ylabel('Predicted Sales ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gru_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.show()

# ========== 7. MAIN EXECUTION ==========
if __name__ == "__main__":
    print("="*60)
    print("🧠 GRU SALES PREDICTION MODEL")
    print("="*60)
    
    # Parameters
    LOOKBACK = 10  # Số tuần trước để dự đoán
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    
    print(f"\n📋 PARAMETERS:")
    print(f"   • Lookback: {LOOKBACK} tuần")
    print(f"   • Feature: Weekly_Sales only")
    print(f"   • Train/Val/Test: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"   • Batch size: {BATCH_SIZE}")
    print(f"   • Epochs: {NUM_EPOCHS}")
    print(f"   • Learning rate: {LEARNING_RATE}")
    
    # 1. Tạo sequential data
    sequences, targets, target_dates = create_sequential_data(df, LOOKBACK)
    print(f"\n✅ Tạo được {len(sequences):,} sequences")
    print(f"   • Sequence shape: {sequences.shape}")
    print(f"   • Target shape: {targets.shape}")
    print(f"   • Target dates: {len(target_dates):,}")
    
    # 2. Split data (80-10-10)
    total_size = len(sequences)
    train_size = int(TRAIN_RATIO * total_size)
    val_size = int(VAL_RATIO * total_size)
    
    # Train set
    train_sequences = sequences[:train_size]
    train_targets = targets[:train_size]
    train_target_dates = target_dates.iloc[:train_size]
    
    # Validation set
    val_sequences = sequences[train_size:train_size + val_size]
    val_targets = targets[train_size:train_size + val_size]
    val_target_dates = target_dates.iloc[train_size:train_size + val_size]
    
    # Test set
    test_sequences = sequences[train_size + val_size:]
    test_targets = targets[train_size + val_size:]
    test_target_dates = target_dates.iloc[train_size + val_size:]
    
    print(f"\n📊 DATA SPLIT:")
    print(f"   • Train: {len(train_sequences):,} samples")
    print(f"   • Validation: {len(val_sequences):,} samples")
    print(f"   • Test: {len(test_sequences):,} samples")
    print(f"   • Test target dates: {len(test_target_dates):,}")
    
    # 3. Scale data
    print(f"\n🔧 Scaling data...")
    
    # Scale sequences
    sequence_scaler = MinMaxScaler()
    train_sequences_scaled = sequence_scaler.fit_transform(train_sequences.reshape(-1, train_sequences.shape[-1])).reshape(train_sequences.shape)
    val_sequences_scaled = sequence_scaler.transform(val_sequences.reshape(-1, val_sequences.shape[-1])).reshape(val_sequences.shape)
    test_sequences_scaled = sequence_scaler.transform(test_sequences.reshape(-1, test_sequences.shape[-1])).reshape(test_sequences.shape)
    
    # Scale targets
    target_scaler = MinMaxScaler()
    train_targets_scaled = target_scaler.fit_transform(train_targets.reshape(-1, 1)).flatten()
    val_targets_scaled = target_scaler.transform(val_targets.reshape(-1, 1)).flatten()
    test_targets_scaled = target_scaler.transform(test_targets.reshape(-1, 1)).flatten()
    
    # 4. Create datasets và dataloaders
    train_dataset = WalmartSequentialDataset(train_sequences_scaled, train_targets_scaled)
    val_dataset = WalmartSequentialDataset(val_sequences_scaled, val_targets_scaled)
    test_dataset = WalmartSequentialDataset(test_sequences_scaled, test_targets_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. Create và train model
    input_size = 1  # Chỉ có 1 feature: Weekly_Sales
    model = GRUSalesPredictor(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2)
    model = model.to(device)
    
    print(f"\n🏗️ MODEL ARCHITECTURE:")
    print(f"   • Input size: {input_size} (Weekly_Sales only)")
    print(f"   • Hidden size: 128")
    print(f"   • Num layers: 2")
    print(f"   • Dropout: 0.2")
    
    # Train model
    model, train_losses, val_losses = train_gru_model(
        model, train_loader, val_loader, 
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE
    )
    
    # 6. Evaluate model
    print(f"\n📊 EVALUATING MODEL...")
    predictions, actuals, rmse, mae, r2 = evaluate_model(model, test_loader, target_scaler)
    
    print(f"\n🏆 TEST RESULTS:")
    print(f"   • RMSE: ${rmse:,.2f}")
    print(f"   • MAE: ${mae:,.2f}")
    print(f"   • R² Score: {r2:.4f}")
    
    # 7. Visualizations
    print(f"\n📈 CREATING VISUALIZATIONS...")
    plot_training_history(train_losses, val_losses)
    plot_predictions(predictions, actuals)
    
    # 8. Save results
    results = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions,
        'actuals': actuals,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    # Save scalers
    import pickle
    with open('sequence_scaler.pkl', 'wb') as f:
        pickle.dump(sequence_scaler, f)
    with open('target_scaler.pkl', 'wb') as f:
        pickle.dump(target_scaler, f)
    
    print(f"\n✅ MODEL TRAINING COMPLETED!")
    print(f"   • Best model saved: best_gru_model.pth")
    print(f"   • Scalers saved: sequence_scaler.pkl, target_scaler.pkl")
    print(f"   • Visualizations saved: gru_training_history.png, gru_predictions_vs_actual.png")
    
    # 9. Summary
    print(f"\n📋 SUMMARY:")
    print(f"   • GRU model đã được train thành công")
    print(f"   • Sử dụng {LOOKBACK} tuần doanh thu trước để dự đoán")
    print(f"   • Feature: Weekly_Sales only")
    print(f"   • Test R²: {r2:.4f}")
    print(f"   • Model sẵn sàng để dự đoán doanh thu dựa trên doanh thu quá khứ")

    # 10. Xuất file dự đoán TEST đúng số dòng (ví dụ 599) để ghép XGBoost
    df_gru_test = pd.DataFrame({
        'date': pd.to_datetime(test_target_dates).values,
        'gru_pred': predictions
    })
    # Gộp theo date (nếu nhiều store trùng ngày) để còn 1 dòng/tuần
    df_gru_test = df_gru_test.groupby('date', as_index=False)['gru_pred'].mean()
    out_path = r'E:\TrainAI\Train\Walmart_new\gru_predictions.csv'
    df_gru_test.to_csv(out_path, index=False)
    print(f"✅ Đã lưu {len(df_gru_test)} dòng test với gru_pred vào: {out_path}")
df_gru_pred = df[['Year', 'WeekOfYear']].copy()
df_gru_pred['gru_pred'] = predictions
df_gru_pred.to_csv('gru_predictions.csv', index=False)

print("✅ Đã lưu gru_predictions.csv")