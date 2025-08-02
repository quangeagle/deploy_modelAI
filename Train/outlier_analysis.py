import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Tạo thư mục output
os.makedirs('report_outputs', exist_ok=True)

# Đọc dữ liệu
df = pd.read_csv("walmart_processed_by_week.csv")

print("="*60)
print("PHÂN TÍCH OUTLIERS BẰNG IQR METHOD")
print("="*60)

# ========== 1. PHÂN TÍCH IQR CHO WEEKLY_SALES ==========
print("\n1. PHÂN TÍCH IQR CHO WEEKLY_SALES")
print("-" * 50)

# Tính Q1, Q3 và IQR
Q1 = df['Weekly_Sales'].quantile(0.25)
Q3 = df['Weekly_Sales'].quantile(0.75)
IQR = Q3 - Q1

# Xác định outliers (giá trị nằm ngoài Q1 - 1.5*IQR và Q3 + 1.5*IQR)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Weekly_Sales'] < lower_bound) | (df['Weekly_Sales'] > upper_bound)]

print(f"📊 Thống kê IQR:")
print(f"  - Q1 (25th percentile): ${Q1:,.2f}")
print(f"  - Q3 (75th percentile): ${Q3:,.2f}")
print(f"  - IQR: ${IQR:,.2f}")
print(f"  - Lower bound: ${lower_bound:,.2f}")
print(f"  - Upper bound: ${upper_bound:,.2f}")
print(f"  - Số lượng outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

# ========== 2. PHÂN TÍCH CHI TIẾT OUTLIERS ==========
print("\n2. PHÂN TÍCH CHI TIẾT OUTLIERS")
print("-" * 50)

if len(outliers) > 0:
    print(f"📈 Thống kê outliers:")
    print(f"  - Outliers thấp nhất: ${outliers['Weekly_Sales'].min():,.2f}")
    print(f"  - Outliers cao nhất: ${outliers['Weekly_Sales'].max():,.2f}")
    print(f"  - Outliers trung bình: ${outliers['Weekly_Sales'].mean():,.2f}")
    
    # Phân tích outliers theo store
    outlier_stores = outliers['Store'].value_counts()
    print(f"\n🏪 Top 5 stores có nhiều outliers nhất:")
    for store, count in outlier_stores.head().items():
        print(f"  - Store {store}: {count} outliers")
    
    # Phân tích outliers theo thời gian
    outlier_months = outliers['Month'].value_counts()
    print(f"\n📅 Tháng có nhiều outliers nhất:")
    for month, count in outlier_months.head().items():
        print(f"  - Tháng {month}: {count} outliers")
    
    # Phân tích outliers theo năm
    outlier_years = outliers['Year'].value_counts()
    print(f"\n📅 Năm có nhiều outliers nhất:")
    for year, count in outlier_years.head().items():
        print(f"  - Năm {year}: {count} outliers")

# ========== 3. PHÂN TÍCH IQR CHO CÁC BIẾN KHÁC ==========
print("\n3. PHÂN TÍCH IQR CHO CÁC BIẾN KHÁC")
print("-" * 50)

numeric_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

for col in numeric_cols:
    print(f"\n📊 Phân tích IQR cho {col}:")
    
    Q1_col = df[col].quantile(0.25)
    Q3_col = df[col].quantile(0.75)
    IQR_col = Q3_col - Q1_col
    
    lower_bound_col = Q1_col - 1.5 * IQR_col
    upper_bound_col = Q3_col + 1.5 * IQR_col
    outliers_col = df[(df[col] < lower_bound_col) | (df[col] > upper_bound_col)]
    
    print(f"  - Q1: {Q1_col:.2f}")
    print(f"  - Q3: {Q3_col:.2f}")
    print(f"  - IQR: {IQR_col:.2f}")
    print(f"  - Lower bound: {lower_bound_col:.2f}")
    print(f"  - Upper bound: {upper_bound_col:.2f}")
    print(f"  - Số outliers: {len(outliers_col)} ({len(outliers_col)/len(df)*100:.1f}%)")

# ========== 4. VẼ BIỂU ĐỒ PHÂN TÍCH ==========
print("\n4. TẠO BIỂU ĐỒ PHÂN TÍCH")
print("-" * 50)

# Biểu đồ 1: Boxplot Weekly Sales
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.boxplot(df['Weekly_Sales'])
plt.title('Boxplot Weekly Sales')
plt.ylabel('Weekly Sales ($)')

# Biểu đồ 2: Histogram với outlier bounds
plt.subplot(2, 3, 2)
plt.hist(df['Weekly_Sales'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(lower_bound, color='red', linestyle='--', label=f'Lower bound: ${lower_bound:,.0f}')
plt.axvline(upper_bound, color='red', linestyle='--', label=f'Upper bound: ${upper_bound:,.0f}')
plt.title('Phân phối Weekly Sales với Outlier Bounds')
plt.xlabel('Weekly Sales ($)')
plt.ylabel('Tần suất')
plt.legend()

# Biểu đồ 3: Outliers theo Store
plt.subplot(2, 3, 3)
if len(outliers) > 0:
    plt.scatter(outliers['Store'], outliers['Weekly_Sales'], color='red', alpha=0.6, s=20)
    plt.title('Outliers theo Store')
    plt.xlabel('Store ID')
    plt.ylabel('Weekly Sales ($)')

# Biểu đồ 4: Outliers theo Tháng
plt.subplot(2, 3, 4)
if len(outliers) > 0:
    plt.scatter(outliers['Month'], outliers['Weekly_Sales'], color='orange', alpha=0.6, s=20)
    plt.title('Outliers theo Tháng')
    plt.xlabel('Tháng')
    plt.ylabel('Weekly Sales ($)')

# Biểu đồ 5: Boxplot cho các biến khác
plt.subplot(2, 3, 5)
df_numeric = df[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']]
plt.boxplot([df_numeric[col] for col in df_numeric.columns], labels=df_numeric.columns)
plt.title('Boxplot các biến khác')
plt.ylabel('Giá trị')
plt.xticks(rotation=45)

# Biểu đồ 6: Scatter plot Weekly Sales vs Temperature với outliers
plt.subplot(2, 3, 6)
plt.scatter(df['Temperature'], df['Weekly_Sales'], alpha=0.5, s=10)
if len(outliers) > 0:
    plt.scatter(outliers['Temperature'], outliers['Weekly_Sales'], color='red', s=30, alpha=0.7, label='Outliers')
    plt.legend()
plt.title('Weekly Sales vs Temperature')
plt.xlabel('Temperature (°F)')
plt.ylabel('Weekly Sales ($)')

plt.tight_layout()
plt.savefig('report_outputs/comprehensive_outlier_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Đã lưu biểu đồ phân tích outliers vào 'report_outputs/comprehensive_outlier_analysis.png'")

# ========== 5. KẾT LUẬN VÀ ĐỀ XUẤT ==========
print("\n5. KẾT LUẬN VÀ ĐỀ XUẤT")
print("-" * 50)

print("📋 Kết luận về outliers:")
print(f"  - Tổng số outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
print(f"  - Phần lớn outliers là doanh số cao (không phải lỗi dữ liệu)")
print(f"  - Outliers có thể do:")
print(f"    + Cửa hàng lớn có hiệu suất cao")
print(f"    + Mùa bán hàng đặc biệt (Black Friday, Christmas)")
print(f"    + Sự kiện khuyến mãi")
print(f"    + Địa điểm cửa hàng tốt")

print("\n💡 Đề xuất xử lý:")
print(f"  - KHÔNG loại bỏ outliers vì chúng có ý nghĩa kinh doanh")
print(f"  - Sử dụng mô hình robust với outliers (Random Forest, XGBoost)")
print(f"  - Thêm features để giải thích outliers")
print(f"  - Phân tích riêng các trường hợp outliers cao")

print("\n" + "="*60)
print("HOÀN THÀNH PHÂN TÍCH OUTLIERS")
print("="*60) 