import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc file CSV
df = pd.read_csv("../Walmart_Sales.csv")

# Tính trung bình các đặc trưng theo từng Store
store_features = df.groupby('Store')[[ 
    'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Weekly_Sales'
]].mean().reset_index()

# Lưu kết quả ra file CSV
store_features.to_csv("store_features_summary.csv", index=False)

# Hiển thị kết quả bảng
print(store_features.head())

# ======== VIZUALIZATION PHÂN TÍCH ĐẶC TRƯNG ========

plt.figure(figsize=(16, 10))

# Vẽ biểu đồ các đặc trưng theo từng Store
features_to_plot = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Weekly_Sales']
for i, feature in enumerate(features_to_plot):
    plt.subplot(2, 3, i + 1)
    sns.barplot(data=store_features, x='Store', y=feature, palette='viridis')
    plt.xticks(rotation=90)
    plt.title(f'{feature} trung bình theo Store')

plt.tight_layout()
plt.show()
