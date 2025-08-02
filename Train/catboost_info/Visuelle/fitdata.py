import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ========================= 1. LOAD DATA =========================
print("📂 Đang load dữ liệu...")

# Load các file dữ liệu
sale = pd.read_csv(r"E:\visuelle2\visuelle2\sales.csv")
price_discount = pd.read_csv(r"E:\visuelle2\visuelle2\price_discount_series.csv")
restocks = pd.read_csv(r"E:\visuelle2\visuelle2\restocks.csv")
customer_data = pd.read_csv(r"E:\visuelle2\visuelle2\customer_data.csv")

print(f"✅ Đã load dữ liệu:")
print(f"   - Sale: {sale.shape}")
print(f"   - Price discount: {price_discount.shape}")
print(f"   - Restocks: {restocks.shape}")
print(f"   - Customer data: {customer_data.shape}")

# ========================= 2. FEATURE ENGINEERING =========================
print("\n🔧 Đang tạo features...")

# --- Tính release_month từ release_date ---
sale['release_date'] = pd.to_datetime(sale['release_date'])
release_min = sale['release_date'].min()
sale['release_month'] = sale['release_date'].apply(
    lambda d: (d.year - release_min.year) * 12 + d.month - release_min.month
)

# --- One-hot encoding cho categorical features ---
categorical_cols = ['category', 'color', 'fabric', 'season']
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = enc.fit_transform(sale[categorical_cols])
encoded_df = pd.DataFrame(
    encoded, 
    columns=enc.get_feature_names_out(categorical_cols),
    index=sale.index
)

# --- Ghép lại với sale ---
sale_features = pd.concat([
    sale[['external_code', 'retail', 'release_month', 'restock', 'image_path']], 
    encoded_df
], axis=1)

# --- Tính avg_discount từ price_discount_series ---
discount_cols = [str(i) for i in range(12)]  # 0, 1, 2, ..., 11
price_discount['avg_discount'] = price_discount[discount_cols].mean(axis=1)
price_discount['max_discount'] = price_discount[discount_cols].max(axis=1)
price_discount['discount_count'] = (price_discount[discount_cols] > 0).sum(axis=1)

# --- Tính tổng restock theo external_code + retail ---
restock_sum = restocks.groupby(['external_code', 'retail'])['qty'].sum().reset_index()
restock_sum.columns = ['external_code', 'retail', 'total_restock_qty']

# --- Đếm số lượng khách hàng mỗi sản phẩm ---
customer_count = customer_data.groupby(['external_code', 'retail']).size().reset_index()
customer_count.columns = ['external_code', 'retail', 'customer_count']

# --- Tính tổng số lượng bán mỗi sản phẩm ---
customer_qty = customer_data.groupby(['external_code', 'retail'])['qty'].sum().reset_index()
customer_qty.columns = ['external_code', 'retail', 'total_sales_qty']

# ========================= 3. MERGE ALL FEATURES =========================
print("🔗 Đang merge features...")

# Merge với price_discount
features = sale_features.merge(
    price_discount[['external_code', 'retail', 'price', 'avg_discount', 'max_discount', 'discount_count'] + discount_cols], 
    on=['external_code', 'retail'], 
    how='left'
)

# Merge với restock_sum
features = features.merge(restock_sum, on=['external_code', 'retail'], how='left')

# Merge với customer_count
features = features.merge(customer_count, on=['external_code', 'retail'], how='left')

# Merge với customer_qty
features = features.merge(customer_qty, on=['external_code', 'retail'], how='left')

# ========================= 4. CLEAN DATA =========================
print("🧹 Đang dọn dữ liệu...")

# Điền giá trị thiếu
features.fillna(0, inplace=True)

# Đảm bảo các cột numeric
numeric_cols = ['release_month', 'restock', 'price', 'avg_discount', 'max_discount', 
                'discount_count', 'total_restock_qty', 'customer_count', 'total_sales_qty'] + discount_cols

for col in numeric_cols:
    if col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)

# ========================= 5. CREATE TARGET =========================
print("🎯 Đang tạo target...")

# Lấy các cột sales (0-11) làm target
target_cols = [str(i) for i in range(12)]
targets = sale[target_cols].copy()

# ========================= 6. FINAL DATASET =========================
print("📊 Tạo dataset cuối cùng...")

# Tách features và targets
X = features.copy()
y = targets.copy()

print(f"\n📈 Dataset cuối cùng:")
print(f"   - Features shape: {X.shape}")
print(f"   - Targets shape: {y.shape}")
print(f"   - Feature columns: {list(X.columns)}")
print(f"   - Target columns: {list(y.columns)}")

# ========================= 7. SAVE DATASET =========================
print("\n💾 Đang lưu dataset...")

# Lưu features và targets
X.to_csv('processed_features.csv', index=False)
y.to_csv('processed_targets.csv', index=False)

print("✅ Đã lưu:")
print("   - processed_features.csv")
print("   - processed_targets.csv")

# ========================= 8. SUMMARY =========================
print("\n📋 Tóm tắt features:")
print(f"   - Categorical features (one-hot): {len([col for col in X.columns if any(cat in col for cat in categorical_cols)])}")
print(f"   - Numeric features: {len(numeric_cols)}")
print(f"   - Discount features: {len(discount_cols)}")
print(f"   - Total features: {X.shape[1]}")

print("\n🎯 Target info:")
print(f"   - Target range: {y.min().min():.4f} - {y.max().max():.4f}")
print(f"   - Target mean: {y.mean().mean():.4f}")
print(f"   - Zero sales ratio: {(y == 0).sum().sum() / (y.shape[0] * y.shape[1]):.2%}")

print("\n✅ Hoàn thành xử lý dữ liệu!")
