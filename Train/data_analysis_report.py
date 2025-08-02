import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pickle
import os

# Tạo thư mục lưu kết quả
os.makedirs('report_outputs', exist_ok=True)

# Đọc dữ liệu
df = pd.read_csv("walmart_processed_by_week.csv")

print("="*60)
print("BÁO CÁO PHÂN TÍCH DỮ LIỆU WALMART SALES")
print("="*60)

# ========== 1. THỐNG KÊ MÔ TẢ ==========
print("\n1. THỐNG KÊ MÔ TẢ DỮ LIỆU")
print("-" * 40)
print("Tổng số bản ghi:", len(df))
print("Số lượng cửa hàng:", df['Store'].nunique())
print("Thời gian dữ liệu:", df['Year'].min(), "-", df['Year'].max())
print("\nThống kê Weekly Sales:")
print(df['Weekly_Sales'].describe())

# ========== 2. PHÂN TÍCH THEO CỬA HÀNG ==========
print("\n2. PHÂN TÍCH THEO CỬA HÀNG")
print("-" * 40)
store_stats = df.groupby('Store')['Weekly_Sales'].agg(['mean', 'std', 'min', 'max'])
print("Top 5 cửa hàng có doanh số cao nhất:")
print(store_stats.sort_values('mean', ascending=False).head())

# ========== 3. PHÂN TÍCH THEO THỜI GIAN ==========
print("\n3. PHÂN TÍCH THEO THỜI GIAN")
print("-" * 40)
monthly_stats = df.groupby('Month')['Weekly_Sales'].mean()
print("Doanh số trung bình theo tháng:")
for month, sales in monthly_stats.items():
    print(f"Tháng {month}: ${sales:,.0f}")

# ========== 4. PHÂN TÍCH NGÀY LỄ ==========
print("\n4. PHÂN TÍCH NGÀY LỄ")
print("-" * 40)
holiday_stats = df.groupby('Holiday_Flag')['Weekly_Sales'].mean()
print(f"Doanh số ngày thường: ${holiday_stats[0]:,.0f}")
print(f"Doanh số ngày lễ: ${holiday_stats[1]:,.0f}")
print(f"Tăng trưởng: {((holiday_stats[1]/holiday_stats[0])-1)*100:.1f}%")

# ========== 5. VẼ BIỂU ĐỒ PHÂN TÍCH ==========
print("\n5. TẠO BIỂU ĐỒ PHÂN TÍCH")
print("-" * 40)

# Biểu đồ 1: Phân phối doanh số
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.hist(df['Weekly_Sales'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Phân phối Weekly Sales')
plt.xlabel('Weekly Sales ($)')
plt.ylabel('Tần suất')

# Biểu đồ 2: Doanh số theo store
plt.subplot(2, 3, 2)
store_sales = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
plt.bar(range(len(store_sales)), store_sales.values, color='lightcoral')
plt.title('Doanh số trung bình theo Store')
plt.xlabel('Store ID')
plt.ylabel('Weekly Sales ($)')
plt.xticks(range(0, len(store_sales), 5))

# Biểu đồ 3: Xu hướng theo thời gian
plt.subplot(2, 3, 3)
time_sales = df.groupby('Week_Index')['Weekly_Sales'].mean()
plt.plot(time_sales.index, time_sales.values, color='green', linewidth=2)
plt.title('Xu hướng Weekly Sales theo thời gian')
plt.xlabel('Week Index')
plt.ylabel('Weekly Sales ($)')

# Biểu đồ 4: Doanh số theo tháng
plt.subplot(2, 3, 4)
month_sales = df.groupby('Month')['Weekly_Sales'].mean()
plt.bar(month_sales.index, month_sales.values, color='gold')
plt.title('Doanh số trung bình theo tháng')
plt.xlabel('Tháng')
plt.ylabel('Weekly Sales ($)')

# Biểu đồ 5: So sánh ngày lễ
plt.subplot(2, 3, 5)
holiday_sales = df.groupby('Holiday_Flag')['Weekly_Sales'].mean()
plt.bar(['Ngày thường', 'Ngày lễ'], holiday_sales.values, color=['lightblue', 'orange'])
plt.title('Doanh số theo ngày lễ')
plt.ylabel('Weekly Sales ($)')

# Biểu đồ 6: Correlation matrix
plt.subplot(2, 3, 6)
numeric_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
correlation_matrix = df[numeric_cols].corr()
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title('Ma trận tương quan')

# Thêm giá trị correlation
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', color='black', fontweight='bold')

plt.tight_layout()
plt.savefig('report_outputs/data_analysis_overview.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Đã lưu biểu đồ tổng quan vào 'report_outputs/data_analysis_overview.png'")

# ========== 6. PHÂN TÍCH CHI TIẾT ==========
print("\n6. PHÂN TÍCH CHI TIẾT")
print("-" * 40)

# Phân tích outliers bằng IQR method
print("\n📊 PHÂN TÍCH OUTLIERS BẰNG IQR METHOD:")
print("-" * 50)

# Tính Q1, Q3 và IQR
Q1 = df['Weekly_Sales'].quantile(0.25)
Q3 = df['Weekly_Sales'].quantile(0.75)
IQR = Q3 - Q1

# Xác định outliers (giá trị nằm ngoài Q1 - 1.5*IQR và Q3 + 1.5*IQR)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Weekly_Sales'] < lower_bound) | (df['Weekly_Sales'] > upper_bound)]

print(f"Q1 (25th percentile): ${Q1:,.2f}")
print(f"Q3 (75th percentile): ${Q3:,.2f}")
print(f"IQR: ${IQR:,.2f}")
print(f"Lower bound: ${lower_bound:,.2f}")
print(f"Upper bound: ${upper_bound:,.2f}")
print(f"Số lượng outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

# Phân tích chi tiết outliers
if len(outliers) > 0:
    print(f"\n📈 Thống kê outliers:")
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

# Vẽ biểu đồ boxplot để visualize outliers
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.boxplot(df['Weekly_Sales'])
plt.title('Boxplot Weekly Sales (Tất cả dữ liệu)')
plt.ylabel('Weekly Sales ($)')

plt.subplot(2, 2, 2)
plt.hist(df['Weekly_Sales'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(lower_bound, color='red', linestyle='--', label=f'Lower bound: ${lower_bound:,.0f}')
plt.axvline(upper_bound, color='red', linestyle='--', label=f'Upper bound: ${upper_bound:,.0f}')
plt.title('Phân phối Weekly Sales với Outlier Bounds')
plt.xlabel('Weekly Sales ($)')
plt.ylabel('Tần suất')
plt.legend()

plt.subplot(2, 2, 3)
if len(outliers) > 0:
    plt.scatter(outliers['Store'], outliers['Weekly_Sales'], color='red', alpha=0.6, s=20)
    plt.title('Outliers theo Store')
    plt.xlabel('Store ID')
    plt.ylabel('Weekly Sales ($)')

plt.subplot(2, 2, 4)
if len(outliers) > 0:
    plt.scatter(outliers['Month'], outliers['Weekly_Sales'], color='orange', alpha=0.6, s=20)
    plt.title('Outliers theo Tháng')
    plt.xlabel('Tháng')
    plt.ylabel('Weekly Sales ($)')

plt.tight_layout()
plt.savefig('report_outputs/outlier_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Đã lưu biểu đồ phân tích outliers vào 'report_outputs/outlier_analysis.png'")

# Phân tích correlation
print("\nCorrelation với Weekly_Sales:")
correlations = df[numeric_cols].corr()['Weekly_Sales'].sort_values(ascending=False)
for col, corr in correlations.items():
    if col != 'Weekly_Sales':
        print(f"{col}: {corr:.3f}")

# Phân tích theo năm
yearly_stats = df.groupby('Year')['Weekly_Sales'].mean()
print(f"\nDoanh số trung bình theo năm:")
for year, sales in yearly_stats.items():
    print(f"Năm {year}: ${sales:,.0f}")

print("\n" + "="*60)
print("HOÀN THÀNH PHÂN TÍCH DỮ LIỆU")
print("="*60) 