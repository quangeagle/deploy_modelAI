# Feature Interaction Analysis - Phân tích tương tác đặc trưng
# Kiểm tra và phân tích các feature interactions được đề xuất

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Đọc dữ liệu
print("📊 Đang đọc dữ liệu Walmart...")
df = pd.read_csv("E:\TrainAI\Train\walmart_processed_by_week.csv")
df['Date'] = pd.to_datetime(df['Date'])

print("="*60)
print("🔍 PHÂN TÍCH FEATURE INTERACTIONS")
print("="*60)

# ========== 1. TẠO FEATURE INTERACTIONS ==========
print("\n1. TẠO FEATURE INTERACTIONS")
print("-" * 40)

# 1.1 Holiday × Temperature
df['Holiday_Temperature'] = df['Holiday_Flag'] * df['Temperature']

# 1.2 Holiday × Month
df['Holiday_Month'] = df['Holiday_Flag'] * df['Month']

# 1.3 Unemployment × CPI
df['Unemployment_CPI'] = df['Unemployment'] * df['CPI']

# 1.4 Thêm một số interactions khác
df['Temperature_Month'] = df['Temperature'] * df['Month']
df['Fuel_Price_CPI'] = df['Fuel_Price'] * df['CPI']

print("✅ Đã tạo các feature interactions:")
print("   • Holiday_Temperature")
print("   • Holiday_Month") 
print("   • Unemployment_CPI")
print("   • Temperature_Month")
print("   • Fuel_Price_CPI")

# ========== 2. PHÂN TÍCH HOLIDAY × TEMPERATURE ==========
print("\n2. PHÂN TÍCH HOLIDAY × TEMPERATURE")
print("-" * 40)

# Tạo figure cho Holiday × Temperature analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('🎉🔥 PHÂN TÍCH HOLIDAY × TEMPERATURE', fontsize=16, fontweight='bold')

# 2.1 Scatter plot: Holiday_Temperature vs Sales
axes[0, 0].scatter(df['Holiday_Temperature'], df['Weekly_Sales'], alpha=0.6, s=20)
axes[0, 0].set_xlabel('Holiday × Temperature')
axes[0, 0].set_ylabel('Doanh thu tuần ($)')
axes[0, 0].set_title('Holiday × Temperature vs Doanh thu')
axes[0, 0].grid(True, alpha=0.3)

# Thêm trend line
z = np.polyfit(df['Holiday_Temperature'], df['Weekly_Sales'], 1)
p = np.poly1d(z)
axes[0, 0].plot(df['Holiday_Temperature'], p(df['Holiday_Temperature']), "r--", alpha=0.8)

# 2.2 Boxplot: Doanh thu theo nhóm Holiday × Temperature
df['Holiday_Temp_Category'] = pd.cut(df['Holiday_Temperature'], 
                                     bins=[0, 20, 40, 60, 80, 100], 
                                     labels=['0', '20-40', '40-60', '60-80', '80-100'])

sns.boxplot(data=df, x='Holiday_Temp_Category', y='Weekly_Sales', ax=axes[0, 1])
axes[0, 1].set_title('Doanh thu theo nhóm Holiday × Temperature')
axes[0, 1].tick_params(axis='x', rotation=45)

# 2.3 Phân tích chi tiết
holiday_temp_stats = df.groupby('Holiday_Flag').agg({
    'Temperature': ['mean', 'std'],
    'Weekly_Sales': ['mean', 'std', 'count']
}).round(2)

print("\n📊 THỐNG KÊ HOLIDAY × TEMPERATURE:")
print(holiday_temp_stats)

# 2.4 Correlation analysis
holiday_temp_corr = df['Holiday_Temperature'].corr(df['Weekly_Sales'])
print(f"\n📊 Correlation Holiday × Temperature vs Sales: {holiday_temp_corr:.4f}")

# 2.5 Line plot: Holiday_Temperature theo thời gian
sample_store = df[df['Store'] == 1].sort_values('Date')
axes[1, 0].plot(sample_store['Date'], sample_store['Holiday_Temperature'], 'purple', alpha=0.7)
axes[1, 0].set_xlabel('Thời gian')
axes[1, 0].set_ylabel('Holiday × Temperature')
axes[1, 0].set_title('Holiday × Temperature theo thời gian')

# 2.6 Correlation heatmap
holiday_temp_corr_matrix = df[['Holiday_Temperature', 'Weekly_Sales']].corr()
sns.heatmap(holiday_temp_corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Correlation: Holiday × Temperature vs Sales')

plt.tight_layout()
plt.savefig('holiday_temperature_interaction.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 3. PHÂN TÍCH HOLIDAY × MONTH ==========
print("\n3. PHÂN TÍCH HOLIDAY × MONTH")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('🎉📅 PHÂN TÍCH HOLIDAY × MONTH', fontsize=16, fontweight='bold')

# 3.1 Scatter plot: Holiday_Month vs Sales
axes[0, 0].scatter(df['Holiday_Month'], df['Weekly_Sales'], alpha=0.6, s=20)
axes[0, 0].set_xlabel('Holiday × Month')
axes[0, 0].set_ylabel('Doanh thu tuần ($)')
axes[0, 0].set_title('Holiday × Month vs Doanh thu')
axes[0, 0].grid(True, alpha=0.3)

# 3.2 Bar plot: Doanh thu trung bình theo Holiday × Month
holiday_month_sales = df.groupby('Holiday_Month')['Weekly_Sales'].mean().reset_index()
axes[0, 1].bar(holiday_month_sales['Holiday_Month'], holiday_month_sales['Weekly_Sales'], alpha=0.7)
axes[0, 1].set_xlabel('Holiday × Month')
axes[0, 1].set_ylabel('Doanh thu trung bình ($)')
axes[0, 1].set_title('Doanh thu trung bình theo Holiday × Month')

# 3.3 Phân tích chi tiết theo tháng
holiday_by_month = df[df['Holiday_Flag'] == 1].groupby('Month')['Weekly_Sales'].agg(['mean', 'std', 'count']).round(2)
print("\n📊 DOANH THU NGÀY LỄ THEO THÁNG:")
print(holiday_by_month)

# 3.4 Line plot: Holiday_Month theo thời gian
sample_store = df[df['Store'] == 1].sort_values('Date')
axes[1, 0].plot(sample_store['Date'], sample_store['Holiday_Month'], 'orange', alpha=0.7)
axes[1, 0].set_xlabel('Thời gian')
axes[1, 0].set_ylabel('Holiday × Month')
axes[1, 0].set_title('Holiday × Month theo thời gian')

# 3.5 Correlation heatmap
holiday_month_corr_matrix = df[['Holiday_Month', 'Weekly_Sales']].corr()
sns.heatmap(holiday_month_corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Correlation: Holiday × Month vs Sales')

plt.tight_layout()
plt.savefig('holiday_month_interaction.png', dpi=300, bbox_inches='tight')
plt.show()

# Thống kê correlation
holiday_month_corr = df['Holiday_Month'].corr(df['Weekly_Sales'])
print(f"\n📊 Correlation Holiday × Month vs Sales: {holiday_month_corr:.4f}")

# ========== 4. PHÂN TÍCH UNEMPLOYMENT × CPI ==========
print("\n4. PHÂN TÍCH UNEMPLOYMENT × CPI")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('👥💰 PHÂN TÍCH UNEMPLOYMENT × CPI', fontsize=16, fontweight='bold')

# 4.1 Scatter plot: Unemployment_CPI vs Sales
axes[0, 0].scatter(df['Unemployment_CPI'], df['Weekly_Sales'], alpha=0.6, s=20)
axes[0, 0].set_xlabel('Unemployment × CPI')
axes[0, 0].set_ylabel('Doanh thu tuần ($)')
axes[0, 0].set_title('Unemployment × CPI vs Doanh thu')
axes[0, 0].grid(True, alpha=0.3)

# Thêm trend line
z = np.polyfit(df['Unemployment_CPI'], df['Weekly_Sales'], 1)
p = np.poly1d(z)
axes[0, 0].plot(df['Unemployment_CPI'], p(df['Unemployment_CPI']), "r--", alpha=0.8)

# 4.2 Boxplot: Doanh thu theo nhóm Unemployment_CPI
df['Unemp_CPI_Category'] = pd.cut(df['Unemployment_CPI'], 
                                  bins=[1400, 1500, 1600, 1700, 1800, 1900], 
                                  labels=['1400-1500', '1500-1600', '1600-1700', '1700-1800', '1800-1900'])

sns.boxplot(data=df, x='Unemp_CPI_Category', y='Weekly_Sales', ax=axes[0, 1])
axes[0, 1].set_title('Doanh thu theo nhóm Unemployment × CPI')
axes[0, 1].tick_params(axis='x', rotation=45)

# 4.3 Phân tích chi tiết
unemp_cpi_stats = df.groupby('Unemp_CPI_Category')['Weekly_Sales'].agg(['mean', 'std', 'count']).round(2)
print("\n📊 THỐNG KÊ UNEMPLOYMENT × CPI:")
print(unemp_cpi_stats)

# 4.4 Line plot: Unemployment_CPI theo thời gian
sample_store = df[df['Store'] == 1].sort_values('Date')
axes[1, 0].plot(sample_store['Date'], sample_store['Unemployment_CPI'], 'green', alpha=0.7)
axes[1, 0].set_xlabel('Thời gian')
axes[1, 0].set_ylabel('Unemployment × CPI')
axes[1, 0].set_title('Unemployment × CPI theo thời gian')

# 4.5 Correlation heatmap
unemp_cpi_corr_matrix = df[['Unemployment_CPI', 'Weekly_Sales']].corr()
sns.heatmap(unemp_cpi_corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Correlation: Unemployment × CPI vs Sales')

plt.tight_layout()
plt.savefig('unemployment_cpi_interaction.png', dpi=300, bbox_inches='tight')
plt.show()

# Thống kê correlation
unemp_cpi_corr = df['Unemployment_CPI'].corr(df['Weekly_Sales'])
print(f"\n📊 Correlation Unemployment × CPI vs Sales: {unemp_cpi_corr:.4f}")

# ========== 5. SO SÁNH CORRELATION TẤT CẢ FEATURES ==========
print("\n5. SO SÁNH CORRELATION TẤT CẢ FEATURES")
print("-" * 40)

# Tính correlation cho tất cả features
features = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 
           'Holiday_Temperature', 'Holiday_Month', 'Unemployment_CPI', 'Temperature_Month', 'Fuel_Price_CPI']

correlation_matrix = df[features].corr()

# In ra correlation với Weekly_Sales
print("\n📊 CORRELATION VỚI DOANH THU (Sắp xếp theo độ mạnh):")
sales_corr = correlation_matrix['Weekly_Sales'].sort_values(ascending=False)
for feature, corr in sales_corr.items():
    if feature != 'Weekly_Sales':
        strength = "Mạnh" if abs(corr) > 0.3 else "Yếu" if abs(corr) < 0.1 else "Trung bình"
        direction = "Thuận" if corr > 0 else "Nghịch"
        print(f"   • {feature}: {corr:.4f} ({strength}, {direction})")

# Vẽ correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.3f')
plt.title('🔥 CORRELATION MATRIX - TẤT CẢ FEATURES & INTERACTIONS', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix_with_interactions.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 6. PHÂN TÍCH CHI TIẾT THEO LOGIC ==========
print("\n6. PHÂN TÍCH CHI TIẾT THEO LOGIC")
print("-" * 40)

# 6.1 Holiday × Temperature logic
print("\n🎉🔥 HOLIDAY × TEMPERATURE LOGIC:")
holiday_temp_analysis = df[df['Holiday_Flag'] == 1].groupby(pd.cut(df[df['Holiday_Flag'] == 1]['Temperature'], 
                                                                    bins=[0, 40, 60, 80, 100])).agg({
    'Weekly_Sales': ['mean', 'count']
}).round(2)
print(holiday_temp_analysis)

# 6.2 Holiday × Month logic
print("\n🎉📅 HOLIDAY × MONTH LOGIC:")
holiday_month_analysis = df[df['Holiday_Flag'] == 1].groupby('Month').agg({
    'Weekly_Sales': ['mean', 'count']
}).round(2)
print(holiday_month_analysis)

# 6.3 Unemployment × CPI logic
print("\n👥💰 UNEMPLOYMENT × CPI LOGIC:")
# Tạo categories cho Unemployment và CPI
df['Unemp_Category'] = pd.cut(df['Unemployment'], bins=[0, 6, 8, 10, 15], 
                              labels=['Thấp (<6%)', 'Trung bình (6-8%)', 'Cao (8-10%)', 'Rất cao (>10%)'])
df['CPI_Category'] = pd.cut(df['CPI'], bins=[200, 210, 215, 220, 230], 
                           labels=['Thấp (200-210)', 'Trung bình (210-215)', 'Cao (215-220)', 'Rất cao (220-230)'])

unemp_cpi_analysis = df.groupby(['Unemp_Category', 'CPI_Category'])['Weekly_Sales'].agg(['mean', 'count']).round(2)
print(unemp_cpi_analysis)

# ========== 7. BÁO CÁO TỔNG HỢP ==========
print("\n" + "="*60)
print("📋 BÁO CÁO TỔNG HỢP FEATURE INTERACTIONS")
print("="*60)

print("\n🎯 KẾT QUẢ PHÂN TÍCH:")

# Tìm features có correlation cao nhất
top_features = sales_corr.head(6)  # Top 5 features (không tính Weekly_Sales)
print("\n🏆 TOP 5 FEATURES CÓ CORRELATION CAO NHẤT:")
for i, (feature, corr) in enumerate(top_features.items(), 1):
    print(f"{i}. {feature}: {corr:.4f}")

print("\n✅ KẾT LUẬN:")
print("• Dữ liệu có đủ thông tin để tạo các feature interactions")
print("• Holiday × Month có thể là interaction quan trọng nhất")
print("• Unemployment × CPI có logic kinh tế rõ ràng")
print("• Holiday × Temperature cần phân tích thêm theo mùa")

print("\n📁 CÁC FILE BIỂU ĐỒ ĐÃ TẠO:")
print("• holiday_temperature_interaction.png")
print("• holiday_month_interaction.png")
print("• unemployment_cpi_interaction.png")
print("• correlation_matrix_with_interactions.png")

print("\n🎯 GỢI Ý CHO MÔ HÌNH:")
print("• Sử dụng Holiday × Month làm feature chính")
print("• Thêm Unemployment × CPI cho logic kinh tế")
print("• Xem xét Holiday × Temperature cho mùa vụ")
print("• Test các interactions khác: Temperature × Month, Fuel_Price × CPI")
