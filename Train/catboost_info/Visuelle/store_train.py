import pandas as pd

# Đọc 2 file
df_img = pd.read_csv("image_features_visuelle.csv")
df_store = pd.read_csv(r"E:\visuelle2\visuelle2\stfore_train.csv")  # Sửa lại tên file
 # Đặt đúng tên file bạn lưu

# Kiểm tra cột ghép
assert 'image_path' in df_img.columns
assert 'image_path' in df_store.columns

# Nối 2 bảng theo image_path
df_merged = df_store.merge(df_img, on="image_path", how="left")

# Kiểm tra sau khi nối
print(f"Số mẫu sau khi merge: {len(df_merged)}")
print(f"Số ảnh có đặc trưng ảnh: {df_merged['img_feat_0'].notnull().sum()}")

# Lưu ra file CSV để kiểm tra
df_merged.to_csv("store_train_with_image_features.csv", index=False)
print("✅ Đã lưu file nối store_train với đặc trưng ảnh.")
