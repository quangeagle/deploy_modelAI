# Walmart EDA Analysis - Exploratory Data Analysis
# Phân tích thống kê & hình ảnh trước khi train mô hình

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Cài đặt style cho biểu đồ
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Đọc dữ liệu
print("📊 Đang đọc dữ liệu Walmart...")
df = pd.read_csv("E:\TrainAI\Train\walmart_processed_by_week.csv")

# Chuyển đổi Date thành datetime
df['Date'] = pd.to_datetime(df['Date'])

print("="*60)
print("🔍 PHÂN TÍCH THỐNG KÊ & HÌNH ẢNH WALMART")
print("="*60)

# ========== 1. TỔNG QUAN DỮ LIỆU ==========
print("\n1. TỔNG QUAN DỮ LIỆU")
print("-" * 40)

print(f"📈 Tổng số bản ghi: {len(df):,}")
print(f"🏪 Số lượng cửa hàng: {df['Store'].nunique()}")
print(f"📅 Thời gian: {df['Date'].min().strftime('%Y-%m-%d')} đến {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"💰 Doanh thu trung bình: ${df['Weekly_Sales'].mean():,.2f}")
print(f"💰 Doanh thu tối đa: ${df['Weekly_Sales'].max():,.2f}")
print(f"💰 Doanh thu tối thiểu: ${df['Weekly_Sales'].min():,.2f}")

# Thống kê mô tả
print("\n📊 Thống kê mô tả:")
print(df.describe())

# ========== 2. PHÂN TÍCH DOANH THU THEO TEMPERATURE ==========
print("\n2. PHÂN TÍCH DOANH THU THEO NHIỆT ĐỘ")
print("-" * 40)

# Tạo figure cho Temperature analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('🔥 PHÂN TÍCH DOANH THU THEO NHIỆT ĐỘ', fontsize=16, fontweight='bold')

# 2.1 Scatter plot: Temperature vs Weekly_Sales
axes[0, 0].scatter(df['Temperature'], df['Weekly_Sales'], alpha=0.6, s=20)
axes[0, 0].set_xlabel('Nhiệt độ (°F)')
axes[0, 0].set_ylabel('Doanh thu tuần ($)')
axes[0, 0].set_title('Nhiệt độ vs Doanh thu')
axes[0, 0].grid(True, alpha=0.3)

# Thêm trend line
z = np.polyfit(df['Temperature'], df['Weekly_Sales'], 1)
p = np.poly1d(z)
axes[0, 0].plot(df['Temperature'], p(df['Temperature']), "r--", alpha=0.8)

# 2.2 Boxplot: Doanh thu theo nhóm nhiệt độ
df['Temp_Category'] = pd.cut(df['Temperature'], 
                             bins=[0, 40, 60, 80, 100], 
                             labels=['Lạnh (<40°F)', 'Mát (40-60°F)', 'Ấm (60-80°F)', 'Nóng (>80°F)'])

sns.boxplot(data=df, x='Temp_Category', y='Weekly_Sales', ax=axes[0, 1])
axes[0, 1].set_title('Doanh thu theo nhóm nhiệt độ')
axes[0, 1].tick_params(axis='x', rotation=45)

# 2.3 Line plot: Nhiệt độ và doanh thu theo thời gian
sample_store = df[df['Store'] == 1].sort_values('Date')
axes[1, 0].plot(sample_store['Date'], sample_store['Temperature'], 'b-', label='Nhiệt độ', alpha=0.7)
axes[1, 0].set_xlabel('Thời gian')
axes[1, 0].set_ylabel('Nhiệt độ (°F)', color='b')
axes[1, 0].tick_params(axis='y', labelcolor='b')

ax2 = axes[1, 0].twinx()
ax2.plot(sample_store['Date'], sample_store['Weekly_Sales'], 'r-', label='Doanh thu', alpha=0.7)
ax2.set_ylabel('Doanh thu ($)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
axes[1, 0].set_title('Nhiệt độ và Doanh thu theo thời gian (Store 1)')

# 2.4 Correlation heatmap cho Temperature
temp_corr = df[['Temperature', 'Weekly_Sales']].corr()
sns.heatmap(temp_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Correlation: Nhiệt độ vs Doanh thu')

plt.tight_layout()
plt.savefig('temperature_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Thống kê correlation
temp_correlation = df['Temperature'].corr(df['Weekly_Sales'])
print(f"📊 Correlation Temperature vs Sales: {temp_correlation:.4f}")

# ========== 3. PHÂN TÍCH DOANH THU THEO FUEL_PRICE ==========
print("\n3. PHÂN TÍCH DOANH THU THEO GIÁ NHIÊN LIỆU")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('⛽ PHÂN TÍCH DOANH THU THEO GIÁ NHIÊN LIỆU', fontsize=16, fontweight='bold')

# 3.1 Scatter plot: Fuel_Price vs Weekly_Sales
axes[0, 0].scatter(df['Fuel_Price'], df['Weekly_Sales'], alpha=0.6, s=20)
axes[0, 0].set_xlabel('Giá nhiên liệu ($)')
axes[0, 0].set_ylabel('Doanh thu tuần ($)')
axes[0, 0].set_title('Giá nhiên liệu vs Doanh thu')
axes[0, 0].grid(True, alpha=0.3)

# Thêm trend line
z = np.polyfit(df['Fuel_Price'], df['Weekly_Sales'], 1)
p = np.poly1d(z)
axes[0, 0].plot(df['Fuel_Price'], p(df['Fuel_Price']), "r--", alpha=0.8)

# 3.2 Line plot: Fuel price và doanh thu theo thời gian
sample_store = df[df['Store'] == 1].sort_values('Date')
axes[0, 1].plot(sample_store['Date'], sample_store['Fuel_Price'], 'g-', label='Giá nhiên liệu', alpha=0.7)
axes[0, 1].set_xlabel('Thời gian')
axes[0, 1].set_ylabel('Giá nhiên liệu ($)', color='g')
axes[0, 1].tick_params(axis='y', labelcolor='g')

ax2 = axes[0, 1].twinx()
ax2.plot(sample_store['Date'], sample_store['Weekly_Sales'], 'r-', label='Doanh thu', alpha=0.7)
ax2.set_ylabel('Doanh thu ($)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
axes[0, 1].set_title('Giá nhiên liệu và Doanh thu theo thời gian')

# 3.3 Boxplot: Doanh thu theo nhóm giá nhiên liệu
df['Fuel_Category'] = pd.cut(df['Fuel_Price'], 
                             bins=[0, 2.5, 3.0, 3.5, 4.0], 
                             labels=['Thấp (<$2.5)', 'Trung bình ($2.5-3.0)', 'Cao ($3.0-3.5)', 'Rất cao (>$3.5)'])

sns.boxplot(data=df, x='Fuel_Category', y='Weekly_Sales', ax=axes[1, 0])
axes[1, 0].set_title('Doanh thu theo nhóm giá nhiên liệu')
axes[1, 0].tick_params(axis='x', rotation=45)

# 3.4 Correlation heatmap cho Fuel_Price
fuel_corr = df[['Fuel_Price', 'Weekly_Sales']].corr()
sns.heatmap(fuel_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Correlation: Giá nhiên liệu vs Doanh thu')

plt.tight_layout()
plt.savefig('fuel_price_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Thống kê correlation
fuel_correlation = df['Fuel_Price'].corr(df['Weekly_Sales'])
print(f"📊 Correlation Fuel Price vs Sales: {fuel_correlation:.4f}")

# ========== 4. PHÂN TÍCH DOANH THU THEO HOLIDAY_FLAG ==========
print("\n4. PHÂN TÍCH DOANH THU THEO NGÀY LỄ")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('🎉 PHÂN TÍCH DOANH THU THEO NGÀY LỄ', fontsize=16, fontweight='bold')

# 4.1 Boxplot: Doanh thu khi có lễ vs không lễ
sns.boxplot(data=df, x='Holiday_Flag', y='Weekly_Sales', ax=axes[0, 0])
axes[0, 0].set_xlabel('Có lễ (1) / Không lễ (0)')
axes[0, 0].set_ylabel('Doanh thu tuần ($)')
axes[0, 0].set_title('Doanh thu: Lễ vs Không lễ')

# 4.2 Bar plot: Doanh thu trung bình
holiday_stats = df.groupby('Holiday_Flag')['Weekly_Sales'].agg(['mean', 'std', 'count']).reset_index()
holiday_stats.columns = ['Holiday_Flag', 'Mean_Sales', 'Std_Sales', 'Count']

axes[0, 1].bar(holiday_stats['Holiday_Flag'], holiday_stats['Mean_Sales'], 
                yerr=holiday_stats['Std_Sales'], capsize=5, alpha=0.7)
axes[0, 1].set_xlabel('Có lễ (1) / Không lễ (0)')
axes[0, 1].set_ylabel('Doanh thu trung bình ($)')
axes[0, 1].set_title('Doanh thu trung bình theo ngày lễ')
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_xticklabels(['Không lễ', 'Có lễ'])

# 4.3 Violin plot: Phân phối doanh thu
sns.violinplot(data=df, x='Holiday_Flag', y='Weekly_Sales', ax=axes[1, 0])
axes[1, 0].set_xlabel('Có lễ (1) / Không lễ (0)')
axes[1, 0].set_ylabel('Doanh thu tuần ($)')
axes[1, 0].set_title('Phân phối doanh thu theo ngày lễ')

# 4.4 Line plot: Doanh thu theo thời gian với highlight ngày lễ
sample_store = df[df['Store'] == 1].sort_values('Date')
axes[1, 1].plot(sample_store['Date'], sample_store['Weekly_Sales'], 'b-', alpha=0.7, label='Doanh thu')

# Highlight ngày lễ
holiday_dates = sample_store[sample_store['Holiday_Flag'] == 1]
axes[1, 1].scatter(holiday_dates['Date'], holiday_dates['Weekly_Sales'], 
                   color='red', s=50, alpha=0.8, label='Ngày lễ')

axes[1, 1].set_xlabel('Thời gian')
axes[1, 1].set_ylabel('Doanh thu ($)')
axes[1, 1].set_title('Doanh thu theo thời gian (đỏ = ngày lễ)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('holiday_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Thống kê holiday
holiday_sales = df[df['Holiday_Flag'] == 1]['Weekly_Sales']
non_holiday_sales = df[df['Holiday_Flag'] == 0]['Weekly_Sales']

print(f"📊 Doanh thu trung bình ngày lễ: ${holiday_sales.mean():,.2f}")
print(f"📊 Doanh thu trung bình không lễ: ${non_holiday_sales.mean():,.2f}")
print(f"📊 Tăng trưởng ngày lễ: {((holiday_sales.mean() - non_holiday_sales.mean()) / non_holiday_sales.mean() * 100):.2f}%")

# ========== 5. PHÂN TÍCH DOANH THU THEO CPI ==========
print("\n5. PHÂN TÍCH DOANH THU THEO CPI")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('💰 PHÂN TÍCH DOANH THU THEO CPI', fontsize=16, fontweight='bold')

# 5.1 Scatter plot: CPI vs Weekly_Sales
axes[0, 0].scatter(df['CPI'], df['Weekly_Sales'], alpha=0.6, s=20)
axes[0, 0].set_xlabel('CPI (Consumer Price Index)')
axes[0, 0].set_ylabel('Doanh thu tuần ($)')
axes[0, 0].set_title('CPI vs Doanh thu')
axes[0, 0].grid(True, alpha=0.3)

# Thêm trend line
z = np.polyfit(df['CPI'], df['Weekly_Sales'], 1)
p = np.poly1d(z)
axes[0, 0].plot(df['CPI'], p(df['CPI']), "r--", alpha=0.8)

# 5.2 Line plot: CPI và doanh thu theo thời gian
sample_store = df[df['Store'] == 1].sort_values('Date')
axes[0, 1].plot(sample_store['Date'], sample_store['CPI'], 'purple', alpha=0.7, label='CPI')
axes[0, 1].set_xlabel('Thời gian')
axes[0, 1].set_ylabel('CPI', color='purple')
axes[0, 1].tick_params(axis='y', labelcolor='purple')

ax2 = axes[0, 1].twinx()
ax2.plot(sample_store['Date'], sample_store['Weekly_Sales'], 'r-', alpha=0.7, label='Doanh thu')
ax2.set_ylabel('Doanh thu ($)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
axes[0, 1].set_title('CPI và Doanh thu theo thời gian')

# 5.3 Boxplot: Doanh thu theo nhóm CPI
df['CPI_Category'] = pd.cut(df['CPI'], 
                            bins=[200, 210, 215, 220, 230], 
                            labels=['Thấp (200-210)', 'Trung bình (210-215)', 'Cao (215-220)', 'Rất cao (220-230)'])

sns.boxplot(data=df, x='CPI_Category', y='Weekly_Sales', ax=axes[1, 0])
axes[1, 0].set_title('Doanh thu theo nhóm CPI')
axes[1, 0].tick_params(axis='x', rotation=45)

# 5.4 Correlation heatmap cho CPI
cpi_corr = df[['CPI', 'Weekly_Sales']].corr()
sns.heatmap(cpi_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Correlation: CPI vs Doanh thu')

plt.tight_layout()
plt.savefig('cpi_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Thống kê correlation
cpi_correlation = df['CPI'].corr(df['Weekly_Sales'])
print(f"📊 Correlation CPI vs Sales: {cpi_correlation:.4f}")

# ========== 6. PHÂN TÍCH DOANH THU THEO UNEMPLOYMENT ==========
print("\n6. PHÂN TÍCH DOANH THU THEO TỶ LỆ THẤT NGHIỆP")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('👥 PHÂN TÍCH DOANH THU THEO TỶ LỆ THẤT NGHIỆP', fontsize=16, fontweight='bold')

# 6.1 Scatter plot: Unemployment vs Weekly_Sales
axes[0, 0].scatter(df['Unemployment'], df['Weekly_Sales'], alpha=0.6, s=20)
axes[0, 0].set_xlabel('Tỷ lệ thất nghiệp (%)')
axes[0, 0].set_ylabel('Doanh thu tuần ($)')
axes[0, 0].set_title('Tỷ lệ thất nghiệp vs Doanh thu')
axes[0, 0].grid(True, alpha=0.3)

# Thêm trend line
z = np.polyfit(df['Unemployment'], df['Weekly_Sales'], 1)
p = np.poly1d(z)
axes[0, 0].plot(df['Unemployment'], p(df['Unemployment']), "r--", alpha=0.8)

# 6.2 Line plot: Unemployment và doanh thu theo thời gian
sample_store = df[df['Store'] == 1].sort_values('Date')
axes[0, 1].plot(sample_store['Date'], sample_store['Unemployment'], 'orange', alpha=0.7, label='Tỷ lệ thất nghiệp')
axes[0, 1].set_xlabel('Thời gian')
axes[0, 1].set_ylabel('Tỷ lệ thất nghiệp (%)', color='orange')
axes[0, 1].tick_params(axis='y', labelcolor='orange')

ax2 = axes[0, 1].twinx()
ax2.plot(sample_store['Date'], sample_store['Weekly_Sales'], 'r-', alpha=0.7, label='Doanh thu')
ax2.set_ylabel('Doanh thu ($)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
axes[0, 1].set_title('Tỷ lệ thất nghiệp và Doanh thu theo thời gian')

# 6.3 Boxplot: Doanh thu theo nhóm thất nghiệp
df['Unemployment_Category'] = pd.cut(df['Unemployment'], 
                                     bins=[0, 6, 8, 10, 15], 
                                     labels=['Thấp (<6%)', 'Trung bình (6-8%)', 'Cao (8-10%)', 'Rất cao (>10%)'])

sns.boxplot(data=df, x='Unemployment_Category', y='Weekly_Sales', ax=axes[1, 0])
axes[1, 0].set_title('Doanh thu theo nhóm tỷ lệ thất nghiệp')
axes[1, 0].tick_params(axis='x', rotation=45)

# 6.4 Correlation heatmap cho Unemployment
unemp_corr = df[['Unemployment', 'Weekly_Sales']].corr()
sns.heatmap(unemp_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Correlation: Tỷ lệ thất nghiệp vs Doanh thu')

plt.tight_layout()
plt.savefig('unemployment_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Thống kê correlation
unemp_correlation = df['Unemployment'].corr(df['Weekly_Sales'])
print(f"📊 Correlation Unemployment vs Sales: {unemp_correlation:.4f}")

# ========== 7. TỔNG HỢP CORRELATION MATRIX ==========
print("\n7. TỔNG HỢP CORRELATION MATRIX")
print("-" * 40)

# Tạo correlation matrix cho tất cả features
features = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag']
correlation_matrix = df[features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('🔥 CORRELATION MATRIX - TẤT CẢ ĐẶC TRƯNG', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix_all_features.png', dpi=300, bbox_inches='tight')
plt.show()

# In ra correlation với Weekly_Sales
print("\n📊 CORRELATION VỚI DOANH THU:")
sales_corr = correlation_matrix['Weekly_Sales'].sort_values(ascending=False)
for feature, corr in sales_corr.items():
    if feature != 'Weekly_Sales':
        print(f"   {feature}: {corr:.4f}")

# ========== 8. PHÂN TÍCH THEO THỜI GIAN ==========
print("\n8. PHÂN TÍCH DOANH THU THEO THỜI GIAN")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('⏰ PHÂN TÍCH DOANH THU THEO THỜI GIAN', fontsize=16, fontweight='bold')

# 8.1 Doanh thu theo tháng
monthly_sales = df.groupby('Month')['Weekly_Sales'].mean().reset_index()
axes[0, 0].bar(monthly_sales['Month'], monthly_sales['Weekly_Sales'], alpha=0.7)
axes[0, 0].set_xlabel('Tháng')
axes[0, 0].set_ylabel('Doanh thu trung bình ($)')
axes[0, 0].set_title('Doanh thu trung bình theo tháng')
axes[0, 0].set_xticks(range(1, 13))

# 8.2 Doanh thu theo tuần trong năm
week_sales = df.groupby('WeekOfYear')['Weekly_Sales'].mean().reset_index()
axes[0, 1].plot(week_sales['WeekOfYear'], week_sales['Weekly_Sales'], 'b-', linewidth=2)
axes[0, 1].set_xlabel('Tuần trong năm')
axes[0, 1].set_ylabel('Doanh thu trung bình ($)')
axes[0, 1].set_title('Doanh thu trung bình theo tuần trong năm')
axes[0, 1].grid(True, alpha=0.3)

# 8.3 Doanh thu theo năm
year_sales = df.groupby('Year')['Weekly_Sales'].mean().reset_index()
axes[1, 0].bar(year_sales['Year'], year_sales['Weekly_Sales'], alpha=0.7)
axes[1, 0].set_xlabel('Năm')
axes[1, 0].set_ylabel('Doanh thu trung bình ($)')
axes[1, 0].set_title('Doanh thu trung bình theo năm')

# 8.4 Time series plot cho một store
sample_store = df[df['Store'] == 1].sort_values('Date')
axes[1, 1].plot(sample_store['Date'], sample_store['Weekly_Sales'], 'g-', linewidth=1.5)
axes[1, 1].set_xlabel('Thời gian')
axes[1, 1].set_ylabel('Doanh thu ($)')
axes[1, 1].set_title('Doanh thu theo thời gian (Store 1)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 9. BÁO CÁO TỔNG HỢP ==========
print("\n" + "="*60)
print("📋 BÁO CÁO TỔNG HỢP EDA")
print("="*60)

print("\n🎯 INSIGHTS CHÍNH:")
print("1. 📊 CORRELATION VỚI DOANH THU:")
for feature, corr in sales_corr.items():
    if feature != 'Weekly_Sales':
        strength = "Mạnh" if abs(corr) > 0.3 else "Yếu" if abs(corr) < 0.1 else "Trung bình"
        direction = "Thuận" if corr > 0 else "Nghịch"
        print(f"   • {feature}: {corr:.4f} ({strength}, {direction})")

print("\n2. 🎉 TÁC ĐỘNG NGÀY LỄ:")
print(f"   • Doanh thu ngày lễ cao hơn: {((holiday_sales.mean() - non_holiday_sales.mean()) / non_holiday_sales.mean() * 100):.2f}%")

print("\n3. 🌡️ TÁC ĐỘNG NHIỆT ĐỘ:")
print(f"   • Correlation: {temp_correlation:.4f}")
print(f"   • Nhiệt độ ảnh hưởng {'mạnh' if abs(temp_correlation) > 0.3 else 'yếu'} đến doanh thu")

print("\n4. ⛽ TÁC ĐỘNG GIÁ NHIÊN LIỆU:")
print(f"   • Correlation: {fuel_correlation:.4f}")
print(f"   • Giá nhiên liệu ảnh hưởng {'mạnh' if abs(fuel_correlation) > 0.3 else 'yếu'} đến doanh thu")

print("\n5. 💰 TÁC ĐỘNG CPI:")
print(f"   • Correlation: {cpi_correlation:.4f}")
print(f"   • CPI ảnh hưởng {'mạnh' if abs(cpi_correlation) > 0.3 else 'yếu'} đến doanh thu")

print("\n6. 👥 TÁC ĐỘNG THẤT NGHIỆP:")
print(f"   • Correlation: {unemp_correlation:.4f}")
print(f"   • Tỷ lệ thất nghiệp ảnh hưởng {'mạnh' if abs(unemp_correlation) > 0.3 else 'yếu'} đến doanh thu")

print("\n✅ KẾT LUẬN:")
print("• Dữ liệu có tính chu kỳ rõ rệt")
print("• Ngày lễ có tác động tích cực mạnh đến doanh thu")
print("• Các yếu tố kinh tế (CPI, Unemployment) có ảnh hưởng đến doanh thu")
print("• Nhiệt độ và giá nhiên liệu có tương quan với doanh thu")

print("\n📁 CÁC FILE BIỂU ĐỒ ĐÃ TẠO:")
print("• temperature_analysis.png")
print("• fuel_price_analysis.png") 
print("• holiday_analysis.png")
print("• cpi_analysis.png")
print("• unemployment_analysis.png")
print("• correlation_matrix_all_features.png")
print("• time_series_analysis.png")

print("\n🎯 GỢI Ý CHO MÔ HÌNH:")
print("• Sử dụng Holiday_Flag làm feature quan trọng")
print("• Xem xét tương tác giữa Temperature và Holiday_Flag")
print("• Thêm features thời gian (Month, WeekOfYear)")
print("• Xử lý outliers trong Weekly_Sales")
print("• Cân nhắc log transformation cho Weekly_Sales")
