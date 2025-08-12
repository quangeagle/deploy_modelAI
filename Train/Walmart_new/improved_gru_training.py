# Improved GRU Sales Prediction Model
# Tinh chỉnh để dự đoán tốt hơn cho các xu hướng khác nhau

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

# ========== 1. IMPROVED GRU MODEL ==========
class ImprovedGRUSalesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super(ImprovedGRUSalesPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers với attention
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional để capture patterns tốt hơn
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # Bidirectional
            num_heads=8,
            dropout=dropout
        )
        
        # Additional layers (TỐI ƯU)
        self.fc1 = nn.Linear(hidden_size * 4, hidden_size)  # 4 vì combined = avg_pool + max_pool
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # GRU processing
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_size * 2)
        
        # Apply layer normalization
        gru_out = self.layer_norm(gru_out)
        
        # Self-attention mechanism
        gru_out = gru_out.transpose(0, 1)  # (seq_len, batch_size, hidden_size * 2)
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        attn_out = attn_out.transpose(0, 1)  # (batch_size, seq_len, hidden_size * 2)
        
        # Global average pooling + max pooling
        avg_pool = torch.mean(attn_out, dim=1)
        max_pool, _ = torch.max(attn_out, dim=1)
        
        # Combine pooling results
        combined = torch.cat([avg_pool, max_pool], dim=1)
        
        # Fully connected layers
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out

# ========== 2. ENHANCED DATASET CREATION ==========
def create_balanced_sequential_data(df, lookback=10):
    """
    Tạo balanced dataset với các xu hướng khác nhau
    """
    print(f"\n🔄 Tạo balanced sequential data với lookback={lookback} tuần...")
    
    all_sequences = []
    all_targets = []
    trend_labels = []  # Để track xu hướng
    target_dates = []  # Ngày tương ứng với mỗi target (NaT cho dữ liệu synthetic)
    target_store_ids = []  # Store tương ứng (None cho synthetic)
    
    # 1. Real data từ dataset
    for store_id in df['Store'].unique():
        store_df = df[df['Store'] == store_id].sort_values('Date')
        sales_data = store_df['Weekly_Sales'].values.astype(np.float32)
        
        for i in range(len(sales_data) - lookback):
            sequence = sales_data[i:i+lookback]
            target = sales_data[i+lookback]
            
            # Determine trend
            if sequence[-1] > sequence[0]:
                trend = "increasing"
            elif sequence[-1] < sequence[0]:
                trend = "decreasing"
            else:
                trend = "stable"
            
            all_sequences.append(sequence.reshape(-1, 1))
            all_targets.append(target)
            trend_labels.append(trend)
            # Lưu ngày & store của tuần target (thực)
            target_dates.append(store_df['Date'].iloc[i+lookback])
            target_store_ids.append(store_id)
    
    # 2. Synthetic data để balance dataset (GIẢM SỐ LƯỢNG)
    print("📊 Tạo synthetic data để balance dataset...")
    
    # Increasing trends (GIẢM TỪ 1000 XUỐNG 200)
    for _ in range(200):
        start_value = np.random.uniform(500000, 1500000)
        sequence = []
        for i in range(lookback):
            # Tạo growth rate realistic hơn
            growth_rate = np.random.uniform(0.01, 0.08)  # Giảm từ 2-15% xuống 1-8%
            if i == 0:
                sequence.append(start_value)
            else:
                sequence.append(sequence[-1] * (1 + growth_rate))
        
        target = sequence[-1] * (1 + np.random.uniform(0.01, 0.08))
        
        all_sequences.append(np.array(sequence).reshape(-1, 1))
        all_targets.append(target)
        trend_labels.append("increasing")
        target_dates.append(pd.NaT)
        target_store_ids.append(None)
    
    # Decreasing trends (GIẢM TỪ 1000 XUỐNG 200)
    for _ in range(200):
        start_value = np.random.uniform(800000, 2000000)
        sequence = []
        for i in range(lookback):
            # Tạo decline rate realistic hơn
            decline_rate = np.random.uniform(0.01, 0.10)  # Giảm từ 2-20% xuống 1-10%
            if i == 0:
                sequence.append(start_value)
            else:
                sequence.append(sequence[-1] * (1 - decline_rate))
        
        target = sequence[-1] * (1 - np.random.uniform(0.01, 0.10))
        
        all_sequences.append(np.array(sequence).reshape(-1, 1))
        all_targets.append(target)
        trend_labels.append("decreasing")
        target_dates.append(pd.NaT)
        target_store_ids.append(None)
    
    # Volatile trends (GIẢM TỪ 500 XUỐNG 100)
    for _ in range(100):
        start_value = np.random.uniform(800000, 1500000)
        sequence = []
        for i in range(lookback):
            if i == 0:
                sequence.append(start_value)
            else:
                # Giảm volatility
                change_rate = np.random.uniform(-0.08, 0.08)  # Giảm từ ±15% xuống ±8%
                sequence.append(sequence[-1] * (1 + change_rate))
        
        # Target based on recent trend
        recent_trend = (sequence[-1] - sequence[-3]) / sequence[-3]
        target = sequence[-1] * (1 + recent_trend * np.random.uniform(0.3, 0.8))  # Giảm multiplier
        
        all_sequences.append(np.array(sequence).reshape(-1, 1))
        all_targets.append(target)
        trend_labels.append("volatile")
        target_dates.append(pd.NaT)
        target_store_ids.append(None)
    
    # Stable trends (GIẢM TỪ 500 XUỐNG 100)
    for _ in range(100):
        base_value = np.random.uniform(800000, 1500000)
        sequence = []
        for i in range(lookback):
            # Giảm variation
            variation = np.random.uniform(-0.03, 0.03)  # Giảm từ ±5% xuống ±3%
            sequence.append(base_value * (1 + variation))
        
        target = base_value * (1 + np.random.uniform(-0.03, 0.03))
        
        all_sequences.append(np.array(sequence).reshape(-1, 1))
        all_targets.append(target)
        trend_labels.append("stable")
        target_dates.append(pd.NaT)
        target_store_ids.append(None)
    
    # Convert to numpy arrays
    all_sequences = np.array(all_sequences)
    all_targets = np.array(all_targets)
    
    print(f"✅ Tạo được {len(all_sequences):,} sequences")
    print(f"   • Increasing: {trend_labels.count('increasing'):,}")
    print(f"   • Decreasing: {trend_labels.count('decreasing'):,}")
    print(f"   • Volatile: {trend_labels.count('volatile'):,}")
    print(f"   • Stable: {trend_labels.count('stable'):,}")
    
    return all_sequences, all_targets, trend_labels, pd.to_datetime(target_dates), pd.Series(target_store_ids)

# ========== 3. ENHANCED TRAINING FUNCTION ==========
def train_improved_gru_model(model, train_loader, val_loader, num_epochs=150, learning_rate=0.001):
    """
    Huấn luyện improved GRU model với enhanced techniques
    """
    print(f"\n🚀 Bắt đầu training Improved GRU model...")
    print(f"📊 Số epochs: {num_epochs}")
    print(f"📚 Learning rate: {learning_rate}")
    
    # Loss function và optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=15, factor=0.5, min_lr=1e-6
    )
    
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
            torch.save(model.state_dict(), 'improved_gru_model.pth')
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping (TĂNG PATIENCE)
        if patience_counter >= 50:  # Tăng từ 25 lên 50
            print(f"⏹️ Early stopping tại epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('improved_gru_model.pth'))
    
    return model, train_losses, val_losses

# ========== 4. TREND VALIDATION FUNCTION ==========
def validate_trend_prediction(input_sequence, prediction, threshold=0.1):
    """
    Validate prediction dựa trên xu hướng input
    """
    # Tính xu hướng của input
    recent_trend = (input_sequence[-1] - input_sequence[-3]) / input_sequence[-3]
    
    # Nếu xu hướng rõ ràng (>10% change)
    if abs(recent_trend) > threshold:
        expected_direction = 1 if recent_trend > 0 else -1
        actual_direction = 1 if prediction > input_sequence[-1] else -1
        
        # Nếu prediction ngược với xu hướng
        if expected_direction != actual_direction:
            # Adjust prediction
            if expected_direction > 0:  # Increasing trend
                adjusted_prediction = input_sequence[-1] * (1 + abs(recent_trend) * 0.8)
            else:  # Decreasing trend
                adjusted_prediction = input_sequence[-1] * (1 - abs(recent_trend) * 0.8)
            
            return adjusted_prediction, True  # True = adjusted
    
    return prediction, False

# ========== 5. MAIN EXECUTION ==========
if __name__ == "__main__":
    print("="*60)
    print("🧠 IMPROVED GRU SALES PREDICTION MODEL")
    print("="*60)
    
    # Parameters
    LOOKBACK = 10
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    BATCH_SIZE = 32  # Giảm batch size để học tốt hơn
    NUM_EPOCHS = 200  # Tăng epochs
    LEARNING_RATE = 0.0005  # Giảm learning rate để học chậm hơn
    
    print(f"\n📋 PARAMETERS:")
    print(f"   • Lookback: {LOOKBACK} tuần")
    print(f"   • Train/Val/Test: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"   • Batch size: {BATCH_SIZE}")
    print(f"   • Epochs: {NUM_EPOCHS}")
    print(f"   • Learning rate: {LEARNING_RATE}")
    
    # 1. Load data
    print("\n📊 Đang đọc dữ liệu Walmart...")
    df = pd.read_csv("../walmart_processed_by_week.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Tạo balanced dataset
    sequences, targets, trend_labels, target_dates, target_store_ids = create_balanced_sequential_data(df, LOOKBACK)
    
    # 3. Split data (REAL-ONLY for val/test; add synthetic to train)
    is_real = ~pd.isna(target_dates)
    real_indices = np.where(is_real.values if hasattr(is_real, 'values') else is_real)[0]
    synthetic_indices = np.where(~(is_real.values if hasattr(is_real, 'values') else is_real))[0]

    num_real = len(real_indices)
    total_size = len(sequences)

    real_train_size = int(TRAIN_RATIO * num_real)
    real_val_size = int(VAL_RATIO * num_real)
    real_test_size = num_real - real_train_size - real_val_size

    # Giữ thứ tự thời gian theo cách đã xây dựng (cuối = mới nhất)
    train_real_idx = real_indices[:real_train_size]
    val_real_idx = real_indices[real_train_size:real_train_size + real_val_size]
    test_real_idx = real_indices[real_train_size + real_val_size:]

    # Train = real train + toàn bộ synthetic (tăng dữ liệu huấn luyện)
    train_idx = np.concatenate([train_real_idx, synthetic_indices]) if len(synthetic_indices) > 0 else train_real_idx

    train_sequences = sequences[train_idx]
    train_targets = targets[train_idx]
    val_sequences = sequences[val_real_idx]
    val_targets = targets[val_real_idx]
    test_sequences = sequences[test_real_idx]
    test_targets = targets[test_real_idx]

    print(f"\n📊 DATA SPLIT (REAL/SYNTHETIC):")
    print(f"   • Total sequences: {total_size:,} (real: {num_real:,}, synthetic: {total_size - num_real:,})")
    print(f"   • Train: {len(train_sequences):,} samples (real: {len(train_real_idx):,}, synthetic: {len(synthetic_indices):,})")
    print(f"   • Validation (real only): {len(val_sequences):,} samples")
    print(f"   • Test (real only): {len(test_sequences):,} samples")
    
    # 4. Scale data
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
    
    # 5. Create datasets và dataloaders
    class WalmartSequentialDataset(Dataset):
        def __init__(self, sequences, targets):
            self.sequences = sequences
            self.targets = targets
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)
    
    train_dataset = WalmartSequentialDataset(train_sequences_scaled, train_targets_scaled)
    val_dataset = WalmartSequentialDataset(val_sequences_scaled, val_targets_scaled)
    test_dataset = WalmartSequentialDataset(test_sequences_scaled, test_targets_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 6. Create và train improved model (TỐI ƯU ARCHITECTURE)
    input_size = 1
    model = ImprovedGRUSalesPredictor(
        input_size=input_size, 
        hidden_size=128,  # Giảm từ 256 xuống 128
        num_layers=2,     # Giảm từ 3 xuống 2
        dropout=0.2       # Giảm từ 0.3 xuống 0.2
    )
    model = model.to(device)
    
    print(f"\n🏗️ OPTIMIZED MODEL ARCHITECTURE:")
    print(f"   • Input size: {input_size}")
    print(f"   • Hidden size: 128")
    print(f"   • Num layers: 2")
    print(f"   • Dropout: 0.2")
    print(f"   • Bidirectional: True")
    print(f"   • Attention: True")
    print(f"   • Synthetic data: Reduced (600 samples)")
    print(f"   • Training: Longer (200 epochs, patience=50)")
    
    # Train model
    model, train_losses, val_losses = train_improved_gru_model(
        model, train_loader, val_loader, 
        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE
    )
    
    # 7. Evaluate model
    print(f"\n📊 EVALUATING IMPROVED MODEL...")
    
    def evaluate_improved_model(model, test_loader, scaler):
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
        
        # Inverse transform
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        return predictions, actuals, rmse, mae, r2
    
    predictions, actuals, rmse, mae, r2 = evaluate_improved_model(model, test_loader, target_scaler)
    
    print(f"\n🏆 IMPROVED MODEL RESULTS:")
    print(f"   • RMSE: ${rmse:,.2f}")
    print(f"   • MAE: ${mae:,.2f}")
    print(f"   • R² Score: {r2:.4f}")
    
    # 8. Save improved model và scalers
    import pickle
    with open('improved_sequence_scaler.pkl', 'wb') as f:
        pickle.dump(sequence_scaler, f)
    with open('improved_target_scaler.pkl', 'wb') as f:
        pickle.dump(target_scaler, f)
    
    print(f"\n✅ IMPROVED MODEL TRAINING COMPLETED!")
    print(f"   • Improved model saved: improved_gru_model.pth")
    print(f"   • Improved scalers saved: improved_sequence_scaler.pkl, improved_target_scaler.pkl")
    
    # 9. Test trend validation
    print(f"\n🧪 TESTING TREND VALIDATION...")
    
    # Test decreasing trend
    decreasing_sequence = [1450000, 1400000, 1350000, 1300000, 1250000, 
                          1200000, 1150000, 1100000, 1000000, 900000]
    
    # Scale sequence
    sequence_scaled = sequence_scaler.transform(np.array(decreasing_sequence).reshape(-1, 1)).reshape(1, -1, 1)
    sequence_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).to(device)
    
    # Predict
    with torch.no_grad():
        prediction_scaled = model(sequence_tensor)
        prediction = target_scaler.inverse_transform(prediction_scaled.cpu().numpy().reshape(-1, 1))[0, 0]
    
    # Validate trend
    adjusted_prediction, was_adjusted = validate_trend_prediction(decreasing_sequence, prediction)
    
    print(f"📊 Decreasing Trend Test:")
    print(f"   • Input: {decreasing_sequence[-3:]}...")
    print(f"   • Raw Prediction: ${prediction:,.2f}")
    print(f"   • Adjusted Prediction: ${adjusted_prediction:,.2f}")
    print(f"   • Was Adjusted: {was_adjusted}")
    
    print(f"\n📋 OPTIMIZED SUMMARY:")
    print(f"   • Optimized GRU model đã được train thành công")
    print(f"   • Reduced synthetic data (600 samples thay vì 3000)")
    print(f"   • Simplified architecture (128 hidden, 2 layers)")
    print(f"   • Longer training (200 epochs, patience=50)")
    print(f"   • Better learning rate (0.0005)")
    print(f"   • Test R²: {r2:.4f}")
    print(f"   • Expected improvement: Higher R² score")
    # Xuất đúng các dòng TEST thực tế (không synthetic) với cột date + gru_pred, 1 dòng/sequence test
    test_target_dates = pd.to_datetime(target_dates[test_real_idx])
    df_gru_test = pd.DataFrame({'date': test_target_dates, 'gru_pred': predictions})
    # Không gộp theo date để giữ chính xác số samples test (ví dụ 599)
    out_path = r'E:\TrainAI\Train\Walmart_new\gru_predictions.csv'
    df_gru_test.to_csv(out_path, index=False)
    print(f"✅ Đã lưu {len(df_gru_test)} dòng test với gru_pred vào: {out_path}")

    # 11. Gắn cột dự đoán vào walmart_processed_by_week.csv (những dòng không phải test = NaN)
    df_with_pred = df.copy()
    df_with_pred['Date'] = pd.to_datetime(df_with_pred['Date'])
    # Lấy store cho từng prediction test
    test_store_ids = target_store_ids[test_real_idx]
    df_test_pred = pd.DataFrame({
        'Date': test_target_dates,
        'Store': test_store_ids.values,
        'gru_pred': predictions
    })
    # Merge theo (Store, Date) để đúng từng cửa hàng/tuần
    df_with_pred = df_with_pred.merge(df_test_pred, on=['Store', 'Date'], how='left')
    out_walmart_with_pred = r'E:\TrainAI\Train\walmart_processed_by_week_with_gru_pred.csv'
    df_with_pred.to_csv(out_walmart_with_pred, index=False)
    print(f"✅ Đã lưu file đầy đủ với cột gru_pred (NaN ở các dòng không test): {out_walmart_with_pred}")