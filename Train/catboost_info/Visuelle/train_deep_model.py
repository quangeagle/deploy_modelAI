import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ========================= 1. LOAD DATA =========================
print("📂 Đang load dữ liệu...")

# Load business features và targets
business_features = pd.read_csv('processed_features.csv')
targets = pd.read_csv('processed_targets.csv')

# Load image features
image_features = pd.read_csv('image_features_visuelle.csv')

print(f"✅ Đã load dữ liệu:")
print(f"   - Business features: {business_features.shape}")
print(f"   - Image features: {image_features.shape}")
print(f"   - Targets: {targets.shape}")

# ========================= 2. MERGE IMAGE + BUSINESS FEATURES =========================
print("\n🔗 Đang merge image và business features...")

# Merge theo image_path
merged_data = business_features.merge(
    image_features, 
    left_on='image_path', 
    right_on='image_path', 
    how='inner'
)

print(f"✅ Sau khi merge: {merged_data.shape}")

# Tách features và targets
X = merged_data.drop(['external_code', 'retail', 'image_path'], axis=1)
y = targets.loc[merged_data.index]

print(f"📊 Dataset cuối cùng:")
print(f"   - Features: {X.shape}")
print(f"   - Targets: {y.shape}")

# Kiểm tra số chiều business features
business_feat_cols = [col for col in X.columns if not col.startswith('img_feat_')]
print(f"   - Business features: {len(business_feat_cols)}")
print(f"   - Image features: 512")

# ========================= 3. DATASET CLASS =========================
class MultimodalDataset(Dataset):
    def __init__(self, features, targets, scaler=None):
        self.features = features
        self.targets = targets
        
        # Tách image features và business features
        self.img_features = features[[f'img_feat_{i}' for i in range(512)]].values
        self.business_features = features.drop([f'img_feat_{i}' for i in range(512)], axis=1).values
        
        # In ra số chiều để debug
        print(f"   - Business features shape: {self.business_features.shape}")
        print(f"   - Image features shape: {self.img_features.shape}")
        
        # Scale business features
        if scaler is None:
            self.scaler = StandardScaler()
            self.business_features = self.scaler.fit_transform(self.business_features)
        else:
            self.scaler = scaler
            self.business_features = self.scaler.transform(self.business_features)
        
        # Convert to tensors
        self.img_features = torch.FloatTensor(self.img_features)
        self.business_features = torch.FloatTensor(self.business_features)
        self.targets = torch.FloatTensor(targets.values)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.img_features[idx], self.business_features[idx], self.targets[idx]

# ========================= 4. DEEP LEARNING MODEL =========================
class MultimodalRevenuePredictor(nn.Module):
    def __init__(self, img_feat_dim=512, business_feat_dim=123, hidden_dim=256, output_dim=12):
        super(MultimodalRevenuePredictor, self).__init__()
        
        # Image feature processing
        self.img_encoder = nn.Sequential(
            nn.Linear(img_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Business feature processing
        self.business_encoder = nn.Sequential(
            nn.Linear(business_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, img_feat, business_feat):
        # Encode image features
        img_encoded = self.img_encoder(img_feat)
        
        # Encode business features
        business_encoded = self.business_encoder(business_feat)
        
        # Concatenate and fuse
        combined = torch.cat([img_encoded, business_encoded], dim=1)
        output = self.fusion(combined)
        
        return output

# ========================= 5. TRAIN/TEST SPLIT =========================
print("\n📊 Đang chia train/test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create datasets
train_dataset = MultimodalDataset(X_train, y_train)
test_dataset = MultimodalDataset(X_test, y_test, scaler=train_dataset.scaler)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"✅ Train set: {len(train_dataset)} samples")
print(f"✅ Test set: {len(test_dataset)} samples")

# ========================= 6. TRAIN MODEL =========================
print("\n🚀 Đang train model...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Sử dụng device: {device}")

model = MultimodalRevenuePredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training loop
epochs = 50
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    
    for img_feat, business_feat, targets in train_loader:
        img_feat, business_feat, targets = img_feat.to(device), business_feat.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(img_feat, business_feat)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for img_feat, business_feat, targets in test_loader:
            img_feat, business_feat, targets = img_feat.to(device), business_feat.to(device), targets.to(device)
            outputs = model(img_feat, business_feat)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Calculate metrics
    train_loss /= len(train_loader)
    val_loss /= len(test_loader)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R²: {r2:.4f}")

# ========================= 7. EVALUATION =========================
print("\n📊 Kết quả cuối cùng:")
print(f"   RMSE: {rmse:.4f}")
print(f"   MAE: {mae:.4f}")
print(f"   R²: {r2:.4f}")

# ========================= 8. VISUALIZATION =========================
plt.figure(figsize=(15, 5))

# Loss curves
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Predictions vs Actual (first sample)
plt.subplot(1, 3, 2)
sample_pred = all_preds[0]
sample_actual = all_targets[0]
plt.plot(range(12), sample_actual, 'o-', label='Actual', markersize=8)
plt.plot(range(12), sample_pred, 's-', label='Predicted', markersize=8)
plt.title('Revenue Prediction - Sample 1')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.legend()

# Feature importance (business features)
plt.subplot(1, 3, 3)
business_feat_importance = np.abs(model.business_encoder[0].weight.data.cpu().numpy()).mean(axis=0)
top_features = np.argsort(business_feat_importance)[-10:]  # Top 10 features
feature_names = X.drop([f'img_feat_{i}' for i in range(512)], axis=1).columns[top_features]
plt.barh(range(10), business_feat_importance[top_features])
plt.yticks(range(10), feature_names)
plt.title('Top 10 Business Features')
plt.xlabel('Importance')

plt.tight_layout()
plt.show()

# ========================= 9. SAVE MODEL =========================
print("\n💾 Đang lưu model...")
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': train_dataset.scaler,
    'feature_names': X.columns.tolist()
}, 'multimodal_revenue_model.pth')

print("✅ Đã lưu model: multimodal_revenue_model.pth")

# ========================= 10. PREDICTION FUNCTION =========================
def predict_revenue(img_features, business_features, model, scaler):
    """
    Dự đoán doanh thu cho sản phẩm mới
    """
    model.eval()
    with torch.no_grad():
        # Scale business features
        business_features_scaled = scaler.transform(business_features.reshape(1, -1))
        
        # Convert to tensors
        img_tensor = torch.FloatTensor(img_features).unsqueeze(0)
        business_tensor = torch.FloatTensor(business_features_scaled)
        
        # Predict
        prediction = model(img_tensor, business_tensor)
        return prediction.cpu().numpy()[0]

print("\n🎯 Model đã sẵn sàng để dự đoán!")
print("   Sử dụng hàm predict_revenue() để dự đoán sản phẩm mới") 