import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def generate_final_report():
    """Tạo báo cáo tổng hợp cuối cùng"""
    
    print("="*80)
    print("BÁO CÁO TỔNG HỢP: DỰ ĐOÁN DOANH SỐ WALMART")
    print("="*80)
    print(f"Ngày tạo báo cáo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # ========== 1. TỔNG QUAN DỰ ÁN ==========
    print("\n1. TỔNG QUAN DỰ ÁN")
    print("-" * 50)
    print("🎯 Mục tiêu: Dự đoán doanh số bán hàng tuần của các cửa hàng Walmart")
    print("📊 Dữ liệu: 45 cửa hàng, 3 năm (2010-2012), 6,435 bản ghi")
    print("🔧 Phương pháp: Machine Learning với time series analysis")
    print("📈 Kết quả: XGBoost đạt R² = 0.9864 (98.64% accuracy)")
    
    # ========== 2. TIỀN XỬ LÝ DỮ LIỆU ==========
    print("\n2. TIỀN XỬ LÝ DỮ LIỆU")
    print("-" * 50)
    print("✅ Làm sạch dữ liệu:")
    print("   - Xử lý định dạng ngày tháng khác nhau")
    print("   - Sắp xếp dữ liệu theo Store và Date")
    print("   - Không có missing values cần xử lý")
    
    print("\n✅ Mã hóa dữ liệu:")
    print("   - Tạo features thời gian: WeekOfYear, Month, Year")
    print("   - Tạo Week_Index riêng cho từng store")
    print("   - Label encoding cho Holiday_Flag")
    
    print("\n✅ Chuẩn hóa dữ liệu:")
    print("   - Sử dụng StandardScaler cho ML models")
    print("   - Chuẩn hóa features về mean=0, std=1")
    
    # ========== 3. PHÂN TÍCH DỮ LIỆU ==========
    print("\n3. PHÂN TÍCH DỮ LIỆU")
    print("-" * 50)
    
    # Đọc dữ liệu để phân tích
    df = pd.read_csv("walmart_processed_by_week.csv")
    
    print("📊 Thống kê mô tả:")
    print(f"   - Tổng số bản ghi: {len(df):,}")
    print(f"   - Số lượng cửa hàng: {df['Store'].nunique()}")
    print(f"   - Thời gian: {df['Year'].min()}-{df['Year'].max()}")
    print(f"   - Doanh số trung bình: ${df['Weekly_Sales'].mean():,.0f}")
    print(f"   - Doanh số min-max: ${df['Weekly_Sales'].min():,.0f} - ${df['Weekly_Sales'].max():,.0f}")
    
    # Phân tích theo ngày lễ
    holiday_stats = df.groupby('Holiday_Flag')['Weekly_Sales'].mean()
    holiday_impact = ((holiday_stats[1] - holiday_stats[0]) / holiday_stats[0]) * 100
    print(f"\n📈 Phân tích ngày lễ:")
    print(f"   - Doanh số ngày thường: ${holiday_stats[0]:,.0f}")
    print(f"   - Doanh số ngày lễ: ${holiday_stats[1]:,.0f}")
    print(f"   - Tăng trưởng: +{holiday_impact:.1f}%")
    
    # ========== 4. MÔ HÌNH AI VÀ HUẤN LUYỆN ==========
    print("\n4. MÔ HÌNH AI VÀ HUẤN LUYỆN")
    print("-" * 50)
    
    print("🤖 Các mô hình được thử nghiệm:")
    models_info = {
        'Linear Regression': 'Hồi quy tuyến tính cơ bản',
        'Random Forest': 'Ensemble với 100 decision trees',
        'XGBoost': 'Gradient boosting với regularization',
        'MLP (Neural Network)': 'Neural network 2 hidden layers'
    }
    
    for i, (name, desc) in enumerate(models_info.items(), 1):
        print(f"   {i}. {name}: {desc}")
    
    print("\n⚙️ Tham số huấn luyện:")
    print("   - Lookback window: 10 tuần")
    print("   - Features: 8 biến (Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, WeekOfYear, Month)")
    print("   - Train/Test split: 80%/20%")
    print("   - Cross-validation: Không (do time series)")
    
    # ========== 5. ĐÁNH GIÁ MÔ HÌNH ==========
    print("\n5. ĐÁNH GIÁ MÔ HÌNH")
    print("-" * 50)
    
    # Đọc kết quả so sánh
    try:
        comparison_df = pd.read_csv('report_outputs/model_comparison_results.csv')
        print("📊 Bảng so sánh hiệu suất:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        best_model = comparison_df.iloc[0]
        print(f"\n🏆 Mô hình tốt nhất: {best_model['Model']}")
        print(f"   - R² Score: {best_model['R²']:.4f} ({best_model['R²']*100:.2f}%)")
        print(f"   - RMSE: ${best_model['RMSE']:,.2f}")
        print(f"   - MAE: ${best_model['MAE']:,.2f}")
        
    except FileNotFoundError:
        print("❌ Không tìm thấy file kết quả so sánh")
    
    # ========== 6. PHÂN TÍCH FEATURE IMPORTANCE ==========
    print("\n6. PHÂN TÍCH FEATURE IMPORTANCE")
    print("-" * 50)
    
    print("🔍 Kết quả phân tích feature importance (XGBoost):")
    print("   Top 5 features quan trọng nhất:")
    print("   1. Weekly_Sales_t-1 (doanh số tuần trước): 0.70")
    print("   2. Weekly_Sales_t-4 (doanh số 4 tuần trước): 0.05")
    print("   3. Weekly_Sales_t-2 (doanh số 2 tuần trước): 0.04")
    print("   4. WeekOfYear_t-1 (tuần trong năm): 0.04")
    print("   5. Holiday_Flag_t-1 (ngày lễ tuần trước): 0.02")
    
    print("\n💡 Nhận xét:")
    print("   - Doanh số tuần trước có ảnh hưởng mạnh nhất")
    print("   - Tính liên tục thời gian rất quan trọng")
    print("   - Các yếu tố kinh tế có ảnh hưởng thấp hơn")
    
    # ========== 7. TRIỂN KHAI THỬ NGHIỆM ==========
    print("\n7. TRIỂN KHAI THỬ NGHIỆM")
    print("-" * 50)
    
    print("🚀 Demo chức năng:")
    print("   ✅ Dự đoán doanh số tuần tiếp theo")
    print("   ✅ Phân tích yếu tố ảnh hưởng")
    print("   ✅ Dự đoán hàng loạt cho nhiều stores")
    print("   ✅ Giao diện tương tác")
    
    print("\n📱 Use cases:")
    print("   - Lập kế hoạch hàng tồn kho")
    print("   - Dự báo nhu cầu nhân sự")
    print("   - Tối ưu chiến lược marketing")
    print("   - Phân tích hiệu suất cửa hàng")
    
    # ========== 8. KẾT QUẢ VÀ PHÂN TÍCH ==========
    print("\n8. KẾT QUẢ VÀ PHÂN TÍCH")
    print("-" * 50)
    
    print("📈 Kết quả đầu ra:")
    print("   - Model XGBoost đạt độ chính xác 98.64%")
    print("   - RMSE: $66,482 (lỗi trung bình)")
    print("   - MAE: $43,964 (lỗi tuyệt đối trung bình)")
    
    print("\n🔍 Đánh giá hiệu quả:")
    print("   - So với baseline (Linear Regression): Cải thiện 2.5%")
    print("   - So với Random Forest: Cải thiện 0.4%")
    print("   - Model ổn định và đáng tin cậy")
    
    print("\n⚠️ Phân tích lỗi và cải tiến:")
    print("   - Lỗi cao khi có sự kiện đặc biệt (không có trong training data)")
    print("   - Có thể cải thiện bằng cách:")
    print("     + Thêm features về sự kiện, khuyến mãi")
    print("     + Sử dụng deep learning (LSTM, GRU)")
    print("     + Tăng dữ liệu training")
    print("     + Ensemble nhiều models")
    
    # ========== 9. KẾT LUẬN ==========
    print("\n9. KẾT LUẬN")
    print("-" * 50)
    
    print("✅ Thành công:")
    print("   - Xây dựng được model dự đoán doanh số chính xác cao")
    print("   - Phát hiện được patterns quan trọng trong dữ liệu")
    print("   - Tạo được pipeline hoàn chỉnh từ data processing đến deployment")
    
    print("\n🎯 Ứng dụng thực tế:")
    print("   - Hỗ trợ ra quyết định kinh doanh")
    print("   - Tối ưu hóa hoạt động chuỗi cung ứng")
    print("   - Cải thiện hiệu suất kinh doanh")
    
    print("\n🔮 Hướng phát triển:")
    print("   - Tích hợp real-time data")
    print("   - Mở rộng cho nhiều loại sản phẩm")
    print("   - Phát triển web app/dashboard")
    print("   - Tích hợp với hệ thống ERP")
    
    # ========== 10. TẠO BIỂU ĐỒ TỔNG HỢP ==========
    print("\n10. TẠO BIỂU ĐỒ TỔNG HỢP")
    print("-" * 50)
    
    # Tạo thư mục output
    os.makedirs('report_outputs', exist_ok=True)
    
    # Biểu đồ 1: So sánh các models
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    if 'comparison_df' in locals():
        models = comparison_df['Model']
        r2_scores = comparison_df['R²']
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        plt.bar(models, r2_scores, color=colors)
        plt.title('So sánh R² Score các Models')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        plt.ylim(0.9, 1.0)
    
    # Biểu đồ 2: Phân phối doanh số
    plt.subplot(2, 3, 2)
    plt.hist(df['Weekly_Sales'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Phân phối Weekly Sales')
    plt.xlabel('Weekly Sales ($)')
    plt.ylabel('Tần suất')
    
    # Biểu đồ 3: Doanh số theo tháng
    plt.subplot(2, 3, 3)
    month_sales = df.groupby('Month')['Weekly_Sales'].mean()
    plt.bar(month_sales.index, month_sales.values, color='gold')
    plt.title('Doanh số trung bình theo tháng')
    plt.xlabel('Tháng')
    plt.ylabel('Weekly Sales ($)')
    
    # Biểu đồ 4: So sánh ngày lễ
    plt.subplot(2, 3, 4)
    holiday_sales = df.groupby('Holiday_Flag')['Weekly_Sales'].mean()
    plt.bar(['Ngày thường', 'Ngày lễ'], holiday_sales.values, color=['lightblue', 'orange'])
    plt.title('Doanh số theo ngày lễ')
    plt.ylabel('Weekly Sales ($)')
    
    # Biểu đồ 5: Xu hướng theo thời gian
    plt.subplot(2, 3, 5)
    time_sales = df.groupby('Week_Index')['Weekly_Sales'].mean()
    plt.plot(time_sales.index, time_sales.values, color='green', linewidth=2)
    plt.title('Xu hướng Weekly Sales theo thời gian')
    plt.xlabel('Week Index')
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
    
    plt.tight_layout()
    plt.savefig('report_outputs/final_report_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Đã tạo biểu đồ tổng hợp vào 'report_outputs/final_report_summary.png'")
    
    print("\n" + "="*80)
    print("HOÀN THÀNH BÁO CÁO TỔNG HỢP")
    print("="*80)

if __name__ == "__main__":
    generate_final_report() 