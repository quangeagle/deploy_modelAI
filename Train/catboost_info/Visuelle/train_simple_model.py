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

# ========================= 2. CREATE SIMPLE FEATURES =========================
print("\n🔧 Đang tạo simple features...")

# Merge theo image_path
merged_data = business_features.merge(
    image_features, 
    left_on='image_path', 
    right_on='image_path', 
    how='inner'
)

# Chỉ lấy image features + restock
simple_features = merged_data[['restock'] + [f'img_feat_{i}' for i in range(512)]].copy()

print(f"📊 Simple dataset:")
print(f"   - Features: {simple_features.shape}")
print(f"   - Targets: {targets.loc[merged_data.index].shape}")

# ========================= 3. SIMPLE DATASET CLASS =========================
class SimpleDataset(Dataset):
    def __init__(self, features, targets, scaler=None):
        self.features = features
        self.targets = targets
        
        # Tách restock và image features
        self.restock = features[['restock']].values
        self.img_features = features[[f'img_feat_{i}' for i in range(512)]].values
        
        # Scale restock
        if scaler is None:
            self.scaler = StandardScaler()
            self.restock = self.scaler.fit_transform(self.restock)
        else:
            self.scaler = scaler
            self.restock = self.scaler.transform(self.restock)
        
        # Convert to tensors
        self.restock = torch.FloatTensor(self.restock)
        self.img_features = torch.FloatTensor(self.img_features)
        self.targets = torch.FloatTensor(targets.values)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.img_features[idx], self.restock[idx], self.targets[idx]

# ========================= 4. SIMPLE MODEL =========================
class SimpleRevenuePredictor(nn.Module):
    def __init__(self, img_feat_dim=512, restock_dim=1, hidden_dim=256, output_dim=12):
        super(SimpleRevenuePredictor, self).__init__()
        
        # Image feature processing
        self.img_encoder = nn.Sequential(
            nn.Linear(img_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Restock processing
        self.restock_encoder = nn.Sequential(
            nn.Linear(restock_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, img_feat, restock_feat):
        # Encode image features
        img_encoded = self.img_encoder(img_feat)
        
        # Encode restock features
        restock_encoded = self.restock_encoder(restock_feat)
        
        # Concatenate and fuse
        combined = torch.cat([img_encoded, restock_encoded], dim=1)
        output = self.fusion(combined)
        
        return output

# ========================= 5. TRAIN/TEST SPLIT =========================
print("\n📊 Đang chia train/test...")

X_simple = simple_features
y_simple = targets.loc[merged_data.index]

X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

# Create datasets
train_dataset = SimpleDataset(X_train, y_train)
test_dataset = SimpleDataset(X_test, y_test, scaler=train_dataset.scaler)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"✅ Train set: {len(train_dataset)} samples")
print(f"✅ Test set: {len(test_dataset)} samples")

# ========================= 6. TRAIN SIMPLE MODEL =========================
print("\n🚀 Đang train simple model...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Sử dụng device: {device}")

model = SimpleRevenuePredictor().to(device)
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
    
    for img_feat, restock_feat, targets in train_loader:
        img_feat, restock_feat, targets = img_feat.to(device), restock_feat.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(img_feat, restock_feat)
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
        for img_feat, restock_feat, targets in test_loader:
            img_feat, restock_feat, targets = img_feat.to(device), restock_feat.to(device), targets.to(device)
            outputs = model(img_feat, restock_feat)
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

# ========================= 7. COMPARISON =========================
print("\n📊 So sánh kết quả:")
print("=" * 50)
print("SIMPLE MODEL (Image + Restock only):")
print(f"   RMSE: {rmse:.4f}")
print(f"   MAE: {mae:.4f}")
print(f"   R²: {r2:.4f}")
print("=" * 50)
print("FULL MODEL (Image + All Business Features):")
print(f"   RMSE: 1.2276")
print(f"   MAE: 0.8807")
print(f"   R²: 0.4069")
print("=" * 50)

# Tính % thay đổi
rmse_change = ((rmse - 1.2276) / 1.2276) * 100
mae_change = ((mae - 0.8807) / 0.8807) * 100
r2_change = ((r2 - 0.4069) / 0.4069) * 100

print(f"📈 Thay đổi:")
print(f"   RMSE: {rmse_change:+.2f}%")
print(f"   MAE: {mae_change:+.2f}%")
print(f"   R²: {r2_change:+.2f}%")

# ========================= 8. SAVE SIMPLE MODEL =========================
print("\n💾 Đang lưu simple model...")
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': train_dataset.scaler,
    'feature_names': ['restock'] + [f'img_feat_{i}' for i in range(512)]
}, 'simple_revenue_model.pth')

print("✅ Đã lưu model: simple_revenue_model.pth")

print("\n🎯 Kết luận:")
if r2 >= 0.4069:
    print("✅ Simple model tốt hơn hoặc bằng full model!")
    print("   → Có thể bỏ business features phức tạp")
else:
    print("❌ Simple model kém hơn full model")
    print("   → Nên giữ lại business features") 