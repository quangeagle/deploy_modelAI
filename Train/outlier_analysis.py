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

# ========== 6. PHÂN TÍCH TẦM QUAN TRỌNG VÀ CHIỀU HƯỚNG ẢNH HƯỞNG ==========
print("\n" + "="*60)
print("PHÂN TÍCH TẦM QUAN TRỌNG VÀ CHIỀU HƯỚNG ẢNH HƯỞNG")
print("="*60)

# ========== 1. PHÂN TÍCH THỐNG KÊ CƠ BẢN ==========
print("\n1. PHÂN TÍCH THỐNG KÊ CƠ BẢN (KHÔNG DÙNG ML)")
print("-" * 50)

# Chọn các features quan trọng
feature_cols = ['Store', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 
                'Month', 'Year', 'Holiday_Flag']

print(f"📊 Phân tích ảnh hưởng của từng feature đến doanh thu:")

# ========== 2. PHÂN TÍCH CORRELATION ĐƠN GIẢN ==========
print("\n2. PHÂN TÍCH CORRELATION VỚI DOANH THU")
print("-" * 50)

# Tính correlation với Weekly_Sales
correlations = {}
for col in feature_cols:
    if col != 'Weekly_Sales':
        corr = df[col].corr(df['Weekly_Sales'])
        correlations[col] = corr
        direction = "Thuận" if corr > 0 else "Nghịch"
        strength = "Mạnh" if abs(corr) > 0.5 else "Trung bình" if abs(corr) > 0.3 else "Yếu"
        print(f"  - {col}: {corr:.4f} ({direction}, {strength})")

# Sắp xếp theo độ mạnh của correlation
sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
print(f"\n🏆 Xếp hạng tầm quan trọng theo correlation:")
for i, (feature, corr) in enumerate(sorted_corr, 1):
    direction = "↗️ Thuận" if corr > 0 else "↘️ Nghịch"
    print(f"  {i}. {feature}: {corr:.4f} {direction}")

# ========== 3. PHÂN TÍCH THEO NHÓM (GROUP ANALYSIS) ==========
print("\n3. PHÂN TÍCH THEO NHÓM")
print("-" * 50)

# Phân tích Holiday Flag
print(f"🎯 Phân tích Holiday Flag:")
holiday_sales = df.groupby('Holiday_Flag')['Weekly_Sales'].agg(['mean', 'count'])
print(f"  - Doanh thu trung bình khi có holiday: ${holiday_sales.loc[1, 'mean']:,.0f}")
print(f"  - Doanh thu trung bình khi không có holiday: ${holiday_sales.loc[0, 'mean']:,.0f}")
holiday_impact = (holiday_sales.loc[1, 'mean'] - holiday_sales.loc[0, 'mean']) / holiday_sales.loc[0, 'mean'] * 100
print(f"  - Ảnh hưởng: {holiday_impact:+.1f}%")

# Phân tích theo Store
print(f"\n🏪 Phân tích theo Store:")
store_analysis = df.groupby('Store')['Weekly_Sales'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
print(f"  - Store có doanh thu cao nhất: Store {store_analysis.index[0]} (${store_analysis.iloc[0]['mean']:,.0f})")
print(f"  - Store có doanh thu thấp nhất: Store {store_analysis.index[-1]} (${store_analysis.iloc[-1]['mean']:,.0f})")
print(f"  - Chênh lệch: {((store_analysis.iloc[0]['mean'] - store_analysis.iloc[-1]['mean']) / store_analysis.iloc[-1]['mean'] * 100):.1f}%")

# Phân tích theo tháng
print(f"\n📅 Phân tích theo tháng:")
monthly_analysis = df.groupby('Month')['Weekly_Sales'].agg(['mean', 'count']).sort_values('mean', ascending=False)
print(f"  - Tháng có doanh thu cao nhất: Tháng {monthly_analysis.index[0]} (${monthly_analysis.iloc[0]['mean']:,.0f})")
print(f"  - Tháng có doanh thu thấp nhất: Tháng {monthly_analysis.index[-1]} (${monthly_analysis.iloc[-1]['mean']:,.0f})")

# ========== 4. PHÂN TÍCH PHÂN VỊ (QUANTILE ANALYSIS) ==========
print("\n4. PHÂN TÍCH PHÂN VỊ")
print("-" * 50)

# Phân tích Weekly_Sales theo phân vị
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
sales_quantiles = df['Weekly_Sales'].quantile(quantiles)
print(f"📈 Phân vị doanh thu:")
for q, value in zip(quantiles, sales_quantiles):
    print(f"  - {q*100:.0f}%: ${value:,.0f}")

# Phân tích features theo phân vị doanh thu
print(f"\n🔍 Phân tích features theo nhóm doanh thu:")
df['Sales_Category'] = pd.cut(df['Weekly_Sales'], 
                              bins=[0, sales_quantiles[0.25], sales_quantiles[0.75], df['Weekly_Sales'].max()],
                              labels=['Thấp', 'Trung bình', 'Cao'])

for feature in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
    feature_by_sales = df.groupby('Sales_Category')[feature].mean()
    print(f"\n  {feature}:")
    print(f"    - Doanh thu thấp: {feature_by_sales['Thấp']:.2f}")
    print(f"    - Doanh thu trung bình: {feature_by_sales['Trung bình']:.2f}")
    print(f"    - Doanh thu cao: {feature_by_sales['Cao']:.2f}")

# ========== 5. PHÂN TÍCH BIẾN THIÊN (VARIANCE ANALYSIS) ==========
print("\n5. PHÂN TÍCH BIẾN THIÊN")
print("-" * 50)

# Tính coefficient of variation (CV) cho mỗi feature
cv_analysis = {}
for feature in feature_cols:
    if feature != 'Weekly_Sales':
        cv = df[feature].std() / df[feature].mean() * 100
        cv_analysis[feature] = cv
        print(f"  - {feature}: CV = {cv:.1f}%")

# Sắp xếp theo CV
sorted_cv = sorted(cv_analysis.items(), key=lambda x: x[1], reverse=True)
print(f"\n📊 Xếp hạng theo độ biến thiên (CV):")
for i, (feature, cv) in enumerate(sorted_cv, 1):
    print(f"  {i}. {feature}: {cv:.1f}%")

# ========== 6. PHÂN TÍCH TƯƠNG QUAN CHÉO ==========
print("\n6. PHÂN TÍCH TƯƠNG QUAN CHÉO")
print("-" * 50)

# Tạo correlation matrix cho tất cả features
correlation_matrix = df[feature_cols + ['Weekly_Sales']].corr()

print(f"🔗 Tương quan giữa các features:")
for i, feature1 in enumerate(feature_cols):
    for j, feature2 in enumerate(feature_cols[i+1:], i+1):
        corr_value = correlation_matrix.loc[feature1, feature2]
        if abs(corr_value) > 0.3:  # Chỉ hiển thị tương quan mạnh
            print(f"  - {feature1} vs {feature2}: {corr_value:.3f}")

# ========== 7. PHÂN TÍCH THEO ĐIỀU KIỆN ==========
print("\n7. PHÂN TÍCH THEO ĐIỀU KIỆN")
print("-" * 50)

# Phân tích khi có/không có holiday
print(f"🎄 Phân tích khi có holiday:")
holiday_data = df[df['Holiday_Flag'] == 1]
non_holiday_data = df[df['Holiday_Flag'] == 0]

print(f"  - Nhiệt độ trung bình khi có holiday: {holiday_data['Temperature'].mean():.1f}°F")
print(f"  - Nhiệt độ trung bình khi không có holiday: {non_holiday_data['Temperature'].mean():.1f}°F")
print(f"  - CPI trung bình khi có holiday: {holiday_data['CPI'].mean():.2f}")
print(f"  - CPI trung bình khi không có holiday: {non_holiday_data['CPI'].mean():.2f}")

# Phân tích theo mùa (nhiệt độ)
print(f"\n🌡️ Phân tích theo nhiệt độ:")
temp_quartiles = df['Temperature'].quantile([0.25, 0.5, 0.75])
cold_data = df[df['Temperature'] <= temp_quartiles[0.25]]
warm_data = df[df['Temperature'] >= temp_quartiles[0.75]]

print(f"  - Doanh thu trung bình khi lạnh: ${cold_data['Weekly_Sales'].mean():,.0f}")
print(f"  - Doanh thu trung bình khi ấm: ${warm_data['Weekly_Sales'].mean():,.0f}")
temp_impact = (warm_data['Weekly_Sales'].mean() - cold_data['Weekly_Sales'].mean()) / cold_data['Weekly_Sales'].mean() * 100
print(f"  - Ảnh hưởng nhiệt độ: {temp_impact:+.1f}%")

# ========== 8. VẼ BIỂU ĐỒ PHÂN TÍCH ==========
print("\n8. TẠO BIỂU ĐỒ PHÂN TÍCH")
print("-" * 50)

plt.figure(figsize=(20, 15))

# Biểu đồ 1: Correlation heatmap
plt.subplot(3, 3, 1)
correlation_matrix = df[feature_cols + ['Weekly_Sales']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Biểu đồ 2: Doanh thu theo Holiday Flag
plt.subplot(3, 3, 2)
df.boxplot(column='Weekly_Sales', by='Holiday_Flag', ax=plt.gca())
plt.title('Weekly Sales by Holiday Flag')
plt.suptitle('')

# Biểu đồ 3: Doanh thu theo tháng
plt.subplot(3, 3, 3)
monthly_sales = df.groupby('Month')['Weekly_Sales'].mean()
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, markersize=8)
plt.title('Average Weekly Sales by Month')
plt.xlabel('Month')
plt.ylabel('Average Weekly Sales ($)')
plt.grid(True, alpha=0.3)

# Biểu đồ 4: Scatter plot Temperature vs Sales
plt.subplot(3, 3, 4)
plt.scatter(df['Temperature'], df['Weekly_Sales'], alpha=0.5, s=10)
plt.title('Temperature vs Weekly Sales')
plt.xlabel('Temperature (°F)')
plt.ylabel('Weekly Sales ($)')

# Biểu đồ 5: Scatter plot CPI vs Sales
plt.subplot(3, 3, 5)
plt.scatter(df['CPI'], df['Weekly_Sales'], alpha=0.5, s=10)
plt.title('CPI vs Weekly Sales')
plt.xlabel('CPI')
plt.ylabel('Weekly Sales ($)')

# Biểu đồ 6: Scatter plot Fuel Price vs Sales
plt.subplot(3, 3, 6)
plt.scatter(df['Fuel_Price'], df['Weekly_Sales'], alpha=0.5, s=10)
plt.title('Fuel Price vs Weekly Sales')
plt.xlabel('Fuel Price ($)')
plt.ylabel('Weekly Sales ($)')

# Biểu đồ 7: Top 10 Stores by Sales
plt.subplot(3, 3, 7)
top_stores = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False).head(10)
plt.barh(range(len(top_stores)), top_stores.values)
plt.yticks(range(len(top_stores)), [f'Store {store}' for store in top_stores.index])
plt.title('Top 10 Stores by Average Sales')
plt.xlabel('Average Weekly Sales ($)')

# Biểu đồ 8: Sales distribution
plt.subplot(3, 3, 8)
plt.hist(df['Weekly_Sales'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Weekly Sales Distribution')
plt.xlabel('Weekly Sales ($)')
plt.ylabel('Frequency')

# Biểu đồ 9: Unemployment vs Sales
plt.subplot(3, 3, 9)
plt.scatter(df['Unemployment'], df['Weekly_Sales'], alpha=0.5, s=10)
plt.title('Unemployment vs Weekly Sales')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Weekly Sales ($)')

plt.tight_layout()
plt.savefig('report_outputs/statistical_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Đã lưu biểu đồ phân tích thống kê vào 'report_outputs/statistical_analysis.png'")

# ========== 9. KẾT LUẬN CHI TIẾT ==========
print("\n9. KẾT LUẬN CHI TIẾT")
print("-" * 50)

print("🎯 TẦM QUAN TRỌNG CỦA CÁC FEATURE (theo correlation):")
print(f"  Top 3 features quan trọng nhất:")
for i, (feature, corr) in enumerate(sorted_corr[:3], 1):
    direction = "TĂNG" if corr > 0 else "GIẢM"
    print(f"    {i}. {feature}: {corr:.4f} ({direction} doanh thu)")

print("\n📈 CHIỀU HƯỚNG ẢNH HƯỞNG:")
print(f"  - Holiday Flag: TĂNG doanh thu {holiday_impact:+.1f}%")
print(f"  - Temperature: Ảnh hưởng {temp_impact:+.1f}% (ấm vs lạnh)")
print(f"  - Store: Chênh lệch lớn giữa các cửa hàng")
print(f"  - Month: Có tính mùa vụ rõ rệt")

print("\n💡 INSIGHTS QUAN TRỌNG:")
print(f"  - Holiday có ảnh hưởng tích cực mạnh đến doanh thu")
print(f"  - Nhiệt độ ảnh hưởng theo mùa")
print(f"  - Store ID là yếu tố quan trọng nhất")
print(f"  - CPI và Unemployment có ảnh hưởng tiêu cực")

print("\n🚀 ĐỀ XUẤT CHO MÔ HÌNH:")
print(f"  - Tập trung vào Store-specific features")
print(f"  - Thêm seasonal features (quý, mùa)")
print(f"  - Xử lý đặc biệt cho holiday periods")
print(f"  - Theo dõi economic indicators (CPI, Unemployment)")

print("\n" + "="*60)
print("HOÀN THÀNH PHÂN TÍCH TẦM QUAN TRỌNG VÀ CHIỀU HƯỚNG ẢNH HƯỞNG")
print("="*60) 