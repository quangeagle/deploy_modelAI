import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class WalmartSalesPredictor:
    """Demo class cho việc dự đoán doanh số Walmart"""
    
    def __init__(self, model_path="model_checkpoints/best_model.pkl", 
                 scaler_path="model_checkpoints/scaler.pkl"):
        """Khởi tạo predictor với model đã train"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Đã load model thành công!")
        except FileNotFoundError:
            print("❌ Không tìm thấy model. Vui lòng train model trước!")
            self.model = None
            self.scaler = None
    
    def prepare_input_data(self, store_data, lookback=10):
        """Chuẩn bị dữ liệu input cho prediction"""
        feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
        
        # Lấy 10 tuần gần nhất
        recent_data = store_data[feature_cols].tail(lookback).values.astype(np.float32)
        
        # Flatten thành 1 vector
        input_features = recent_data.flatten().reshape(1, -1)
        
        return input_features
    
    def predict_next_week_sales(self, store_data):
        """Dự đoán doanh số tuần tiếp theo"""
        if self.model is None:
            return None, "Model chưa được load"
        
        try:
            # Chuẩn bị input
            X_input = self.prepare_input_data(store_data)
            
            # Chuẩn hóa
            X_scaled = self.scaler.transform(X_input)
            
            # Dự đoán
            prediction = self.model.predict(X_scaled)[0]
            
            return prediction, "Thành công"
            
        except Exception as e:
            return None, f"Lỗi: {str(e)}"
    
    def demo_prediction(self):
        """Demo dự đoán với dữ liệu mẫu"""
        print("\n" + "="*60)
        print("DEMO DỰ ĐOÁN DOANH SỐ WALMART")
        print("="*60)
        
        # Đọc dữ liệu
        df = pd.read_csv("walmart_processed_by_week.csv")
        
        # Demo cho 3 stores khác nhau
        demo_stores = [1, 15, 30]
        
        for store_id in demo_stores:
            print(f"\n--- Dự đoán cho Store {store_id} ---")
            
            # Lấy dữ liệu của store
            store_data = df[df['Store'] == store_id].sort_values('Week_Index')
            
            if len(store_data) < 10:
                print(f"❌ Store {store_id} không đủ dữ liệu (cần ít nhất 10 tuần)")
                continue
            
            # Lấy thông tin tuần gần nhất
            latest_week = store_data.iloc[-1]
            print(f"Tuần gần nhất: Week {latest_week['Week_Index']}")
            print(f"Doanh số tuần gần nhất: ${latest_week['Weekly_Sales']:,.2f}")
            
            # Dự đoán tuần tiếp theo
            prediction, status = self.predict_next_week_sales(store_data)
            
            if prediction is not None:
                print(f"🔮 Dự đoán tuần tiếp theo: ${prediction:,.2f}")
                
                # Tính % thay đổi
                change_percent = ((prediction - latest_week['Weekly_Sales']) / latest_week['Weekly_Sales']) * 100
                change_text = "tăng" if change_percent > 0 else "giảm"
                print(f"📈 Dự kiến {change_text} {abs(change_percent):.1f}%")
                
                # Hiển thị thông tin thời tiết, kinh tế
                print(f"🌡️ Nhiệt độ: {latest_week['Temperature']:.1f}°F")
                print(f"⛽ Giá xăng: ${latest_week['Fuel_Price']:.2f}")
                print(f"💰 CPI: {latest_week['CPI']:.2f}")
                print(f"👥 Tỷ lệ thất nghiệp: {latest_week['Unemployment']:.2f}%")
                
                # Phân tích yếu tố ảnh hưởng
                self.analyze_factors(store_data)
                
            else:
                print(f"❌ {status}")
    
    def analyze_factors(self, store_data):
        """Phân tích các yếu tố ảnh hưởng"""
        print("\n📊 Phân tích yếu tố ảnh hưởng:")
        
        # Tính correlation với Weekly_Sales
        numeric_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        correlations = store_data[numeric_cols].corr()['Weekly_Sales'].sort_values(ascending=False)
        
        print("Mối tương quan với doanh số:")
        for col, corr in correlations.items():
            if col != 'Weekly_Sales':
                impact = "Tích cực" if corr > 0 else "Tiêu cực"
                strength = "Mạnh" if abs(corr) > 0.3 else "Yếu" if abs(corr) < 0.1 else "Trung bình"
                print(f"  - {col}: {corr:.3f} ({impact}, {strength})")
        
        # Phân tích theo ngày lễ
        holiday_avg = store_data[store_data['Holiday_Flag'] == 1]['Weekly_Sales'].mean()
        normal_avg = store_data[store_data['Holiday_Flag'] == 0]['Weekly_Sales'].mean()
        
        if not pd.isna(holiday_avg) and not pd.isna(normal_avg):
            holiday_impact = ((holiday_avg - normal_avg) / normal_avg) * 100
            print(f"  - Ngày lễ: Tăng {holiday_impact:.1f}% so với ngày thường")
    
    def batch_prediction_demo(self):
        """Demo dự đoán hàng loạt"""
        print("\n" + "="*60)
        print("DEMO DỰ ĐOÁN HÀNG LOẠT")
        print("="*60)
        
        df = pd.read_csv("walmart_processed_by_week.csv")
        
        # Chọn 5 stores để demo
        demo_stores = df['Store'].unique()[:5]
        predictions = []
        
        for store_id in demo_stores:
            store_data = df[df['Store'] == store_id].sort_values('Week_Index')
            
            if len(store_data) >= 10:
                prediction, _ = self.predict_next_week_sales(store_data)
                if prediction is not None:
                    latest_sales = store_data.iloc[-1]['Weekly_Sales']
                    predictions.append({
                        'Store': store_id,
                        'Current_Sales': latest_sales,
                        'Predicted_Sales': prediction,
                        'Change_Percent': ((prediction - latest_sales) / latest_sales) * 100
                    })
        
        # Tạo bảng kết quả
        if predictions:
            results_df = pd.DataFrame(predictions)
            print("\n📋 Kết quả dự đoán cho 5 stores:")
            print(results_df.to_string(index=False, float_format='%.2f'))
            
            # Vẽ biểu đồ so sánh
            plt.figure(figsize=(12, 6))
            
            x = range(len(predictions))
            current_sales = [p['Current_Sales'] for p in predictions]
            predicted_sales = [p['Predicted_Sales'] for p in predictions]
            
            plt.subplot(1, 2, 1)
            plt.bar([i-0.2 for i in x], current_sales, width=0.4, label='Doanh số hiện tại', color='skyblue')
            plt.bar([i+0.2 for i in x], predicted_sales, width=0.4, label='Doanh số dự đoán', color='orange')
            plt.xlabel('Store ID')
            plt.ylabel('Weekly Sales ($)')
            plt.title('So sánh doanh số hiện tại vs dự đoán')
            plt.legend()
            plt.xticks(x, [p['Store'] for p in predictions])
            
            plt.subplot(1, 2, 2)
            changes = [p['Change_Percent'] for p in predictions]
            colors = ['green' if c > 0 else 'red' for c in changes]
            plt.bar(x, changes, color=colors)
            plt.xlabel('Store ID')
            plt.ylabel('Thay đổi (%)')
            plt.title('Phần trăm thay đổi dự kiến')
            plt.xticks(x, [p['Store'] for p in predictions])
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('report_outputs/batch_prediction_demo.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("\n✅ Đã lưu biểu đồ demo vào 'report_outputs/batch_prediction_demo.png'")
    
    def interactive_demo(self):
        """Demo tương tác với người dùng"""
        print("\n" + "="*60)
        print("DEMO TƯƠNG TÁC")
        print("="*60)
        
        df = pd.read_csv("walmart_processed_by_week.csv")
        
        while True:
            print("\nChọn chức năng:")
            print("1. Dự đoán cho store cụ thể")
            print("2. Dự đoán hàng loạt")
            print("3. Thoát")
            
            choice = input("\nNhập lựa chọn (1-3): ").strip()
            
            if choice == '1':
                try:
                    store_id = int(input("Nhập Store ID (1-45): "))
                    if store_id < 1 or store_id > 45:
                        print("❌ Store ID phải từ 1-45")
                        continue
                    
                    store_data = df[df['Store'] == store_id].sort_values('Week_Index')
                    
                    if len(store_data) < 10:
                        print(f"❌ Store {store_id} không đủ dữ liệu")
                        continue
                    
                    prediction, status = self.predict_next_week_sales(store_data)
                    
                    if prediction is not None:
                        latest_sales = store_data.iloc[-1]['Weekly_Sales']
                        print(f"\n🔮 Dự đoán cho Store {store_id}:")
                        print(f"  - Doanh số hiện tại: ${latest_sales:,.2f}")
                        print(f"  - Doanh số dự đoán: ${prediction:,.2f}")
                        
                        change_percent = ((prediction - latest_sales) / latest_sales) * 100
                        change_text = "tăng" if change_percent > 0 else "giảm"
                        print(f"  - Dự kiến {change_text} {abs(change_percent):.1f}%")
                    
                except ValueError:
                    print("❌ Vui lòng nhập số hợp lệ")
                    
            elif choice == '2':
                self.batch_prediction_demo()
                
            elif choice == '3':
                print("👋 Tạm biệt!")
                break
                
            else:
                print("❌ Lựa chọn không hợp lệ")

def main():
    """Hàm main để chạy demo"""
    predictor = WalmartSalesPredictor()
    
    if predictor.model is not None:
        # Chạy các demo
        predictor.demo_prediction()
        predictor.batch_prediction_demo()
        predictor.interactive_demo()
    else:
        print("❌ Không thể chạy demo vì model chưa được load")

if __name__ == "__main__":
    main() 