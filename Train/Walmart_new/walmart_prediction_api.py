import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class WalmartPredictionAPI:
    def __init__(self):
        """Khởi tạo API và load model đã train"""
        self.model = None
        self.feature_columns = None
        self.model_info = None
        self.is_model_loaded = False
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load model đã train từ output_relative"""
        try:
            model_path = 'output_relative/xgb_model.pkl'
            feature_path = 'output_relative/feature_columns.txt'
            info_path = 'output_relative/model_info.txt'
            
            if not os.path.exists(model_path):
                print("❌ Không tìm thấy model. Vui lòng chạy xgboost_stacking_relative_changes.py trước.")
                return False
            
            # Load model
            self.model = joblib.load(model_path)
            print("✅ Đã load XGBoost model thành công!")
            
            # Load feature columns
            with open(feature_path, 'r') as f:
                self.feature_columns = [line.strip() for line in f.readlines()]
            print(f"✅ Đã load {len(self.feature_columns)} features")
            
            # Load model info
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    self.model_info = f.read()
                print("✅ Đã load thông tin model")
            
            self.is_model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi load model: {e}")
            return False
    
    def prepare_input_data(self, weekly_sales_data, external_factors_current, external_factors_previous=None):
        """
        Chuẩn bị dữ liệu đầu vào cho prediction
        
        Args:
            weekly_sales_data: List 10 tuần doanh thu [week1, week2, ..., week10]
            external_factors_current: Dict các yếu tố bên ngoài tuần dự đoán
            external_factors_previous: Dict các yếu tố bên ngoài tuần trước (để tính relative changes)
        """
        try:
            # Tạo DataFrame với 10 tuần doanh thu
            df = pd.DataFrame({
                'Weekly_Sales': weekly_sales_data,
                'WeekOfYear': list(range(1, 11)),
                'Year': [2024] * 10,
                'Month': [1] * 10,  # Có thể cập nhật sau
                'DayOfWeek': [1] * 10  # Thứ 2
            })
            
            # Tính relative changes cho doanh thu
            df['sales_pct_change'] = df['Weekly_Sales'].pct_change().fillna(0)
            df['sales_pct_change_2'] = df['Weekly_Sales'].pct_change(2).fillna(0)
            df['sales_pct_change_3'] = df['Weekly_Sales'].pct_change(3).fillna(0)
            
            # Tính moving averages
            df['sales_ma_3'] = df['Weekly_Sales'].rolling(3).mean().fillna(df['Weekly_Sales'])
            df['sales_ma_5'] = df['Weekly_Sales'].rolling(5).mean().fillna(df['Weekly_Sales'])
            
            # Tính trend
            df['sales_trend'] = df['Weekly_Sales'].diff().fillna(0)
            
            # Tạo features cho tuần dự đoán (tuần 11)
            prediction_features = {}
            
            # Doanh thu features
            prediction_features['sales_pct_change'] = df['sales_pct_change'].iloc[-1]
            prediction_features['sales_pct_change_2'] = df['sales_pct_change_2'].iloc[-1]
            prediction_features['sales_pct_change_3'] = df['sales_pct_change_3'].iloc[-1]
            prediction_features['sales_ma_3'] = df['sales_ma_3'].iloc[-1]
            prediction_features['sales_ma_5'] = df['sales_ma_5'].iloc[-1]
            prediction_features['sales_trend'] = df['sales_trend'].iloc[-1]
            
            # External factors (relative changes so với tuần trước)
            if external_factors_previous is None:
                # Nếu không có dữ liệu tuần trước, sử dụng giá trị mặc định
                print("⚠️ Không có dữ liệu tuần trước, sử dụng giá trị mặc định")
                external_factors_previous = {
                    'Temperature': 25,  # Giá trị mặc định
                    'Fuel_Price': 3.5,
                    'CPI': 250,
                    'Unemployment': 5.0
                }
            
            # Tính relative changes thực tế
            for factor in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
                if factor in external_factors_current and factor in external_factors_previous:
                    current_value = external_factors_current[factor]
                    previous_value = external_factors_previous[factor]
                    
                    # Tính % thay đổi
                    if previous_value != 0:
                        pct_change = (current_value - previous_value) / previous_value
                    else:
                        pct_change = 0  # Tránh chia cho 0
                    
                    # Lưu % thay đổi (đây là feature chính cho model Relative Changes)
                    prediction_features[f'{factor}_change'] = pct_change * 100  # Chuyển thành %
                    
                    # Lưu giá trị tuyệt đối hiện tại
                    prediction_features[f'{factor}'] = current_value
                else:
                    # Nếu thiếu dữ liệu, đặt giá trị mặc định
                    prediction_features[f'{factor}_change'] = 0
                    prediction_features[f'{factor}'] = external_factors_previous.get(factor, 0)
            
            # Holiday flag
            prediction_features['Holiday_Flag'] = external_factors_current.get('Holiday_Flag', 0)
            
            # Time features
            prediction_features['WeekOfYear'] = 11
            prediction_features['Month'] = 1
            prediction_features['Year'] = 2024
            prediction_features['DayOfWeek'] = 1
            prediction_features['Is_Weekend'] = 0
            
            # Tạo DataFrame features
            features_df = pd.DataFrame([prediction_features])
            
            # Đảm bảo có đủ columns như model đã train
            missing_cols = set(self.feature_columns) - set(features_df.columns)
            for col in missing_cols:
                features_df[col] = 0
            
            # Sắp xếp columns theo thứ tự model đã train
            features_df = features_df[self.feature_columns]
            
            return features_df, df
            
        except Exception as e:
            print(f"❌ Lỗi khi chuẩn bị dữ liệu: {e}")
            return None, None
    
    def predict_gru_sales(self, weekly_sales_data):
        """
        Dự đoán doanh thu tuần tiếp theo dựa trên GRU (đơn giản hóa)
        Trong thực tế, bạn sẽ load GRU model đã train
        """
        try:
            # Đơn giản hóa: dựa trên trend và moving average
            recent_sales = weekly_sales_data[-3:]  # 3 tuần gần nhất
            trend = np.mean(np.diff(recent_sales))
            ma_3 = np.mean(recent_sales)
            
            # Dự đoán dựa trên trend và moving average
            gru_prediction = ma_3 + trend
            
            # Đảm bảo dự đoán không âm
            gru_prediction = max(gru_prediction, np.min(recent_sales) * 0.8)
            
            return gru_prediction
            
        except Exception as e:
            print(f"❌ Lỗi khi dự đoán GRU: {e}")
            return None
    
    def predict_xgboost_adjustment(self, features_df):
        """
        Dự đoán điều chỉnh từ XGBoost dựa trên external factors
        """
        try:
            if not self.is_model_loaded:
                print("❌ Model chưa được load")
                return None
            
            # Dự đoán
            xgb_prediction = self.model.predict(features_df)[0]
            
            return xgb_prediction
            
        except Exception as e:
            print(f"❌ Lỗi khi dự đoán XGBoost: {e}")
            return None
    
    def analyze_external_factors(self, features_df, external_factors):
        """
        Phân tích tác động của các yếu tố bên ngoài
        """
        try:
            print(f"🔍 Debug: external_factors keys: {list(external_factors.keys())}")
            print(f"🔍 Debug: external_factors values: {external_factors}")
            analysis = {}
            
            # Phân tích từng factor
            for factor in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
                try:
                    if f'{factor}_change' in features_df.columns:
                        pct_change = features_df[f'{factor}_change'].iloc[0] / 100  # Chuyển từ % về decimal
                        # Lấy giá trị tuyệt đối từ external_factors, không phải từ features_df
                        # external_factors có keys: 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag'
                        absolute_value = external_factors.get(factor, 0)
                        print(f"🔍 Debug {factor}: absolute_value={absolute_value}, pct_change={pct_change}")
                        
                        # Đánh giá tác động
                        if factor == 'Temperature':
                            if pct_change > 0.1:  # Tăng >10%
                                impact = "Nhiệt độ tăng cao → có thể giảm doanh thu (người ít ra ngoài)"
                                direction = "Giảm"
                            elif pct_change < -0.1:  # Giảm >10%
                                impact = "Nhiệt độ giảm → có thể tăng doanh thu (mua sắm mùa đông)"
                                direction = "Tăng"
                            else:
                                impact = "Nhiệt độ ổn định → ít ảnh hưởng"
                                direction = "Trung tính"
                        
                        elif factor == 'Fuel_Price':
                            if pct_change > 0.05:  # Tăng >5%
                                impact = "Giá xăng tăng → giảm sức mua → giảm doanh thu"
                                direction = "Giảm"
                            elif pct_change < -0.05:  # Giảm >5%
                                impact = "Giá xăng giảm → tăng sức mua → tăng doanh thu"
                                direction = "Tăng"
                            else:
                                impact = "Giá xăng ổn định → ít ảnh hưởng"
                                direction = "Trung tính"
                        
                        elif factor == 'CPI':
                            if pct_change > 0.02:  # Tăng >2%
                                impact = "CPI tăng → lạm phát → giảm sức mua → giảm doanh thu"
                                direction = "Giảm"
                            elif pct_change < -0.02:  # Giảm >2%
                                impact = "CPI giảm → giảm lạm phát → tăng sức mua → tăng doanh thu"
                                direction = "Tăng"
                            else:
                                impact = "CPI ổn định → ít ảnh hưởng"
                                direction = "Trung tính"
                        
                        elif factor == 'Unemployment':
                            if pct_change > 0.01:  # Tăng >1%
                                impact = "Tỷ lệ thất nghiệp tăng → giảm sức mua → giảm doanh thu"
                                direction = "Giảm"
                            elif pct_change < -0.01:  # Giảm >1%
                                impact = "Tỷ lệ thất nghiệp giảm → tăng sức mua → tăng doanh thu"
                                direction = "Tăng"
                            else:
                                impact = "Tỷ lệ thất nghiệp ổn định → ít ảnh hưởng"
                                direction = "Trung tính"
                        
                        analysis[factor] = {
                            'value': absolute_value,
                            'pct_change': pct_change,
                            'impact': impact,
                            'direction': direction
                        }
                    else:
                        # Nếu không có data, sử dụng giá trị mặc định
                        analysis[factor] = {
                            'value': 0,
                            'pct_change': 0,
                            'impact': 'Không có dữ liệu',
                            'direction': 'Trung tính'
                        }
                except Exception as e:
                    print(f"⚠️ Lỗi khi phân tích {factor}: {e}")
                    # Sử dụng giá trị mặc định nếu có lỗi
                    analysis[factor] = {
                        'value': 0,
                        'pct_change': 0,
                        'impact': f'Lỗi phân tích: {str(e)}',
                        'direction': 'Trung tính'
                    }
            
            # Holiday flag
            if 'Holiday_Flag' in external_factors:
                holiday = external_factors['Holiday_Flag']
                if holiday == 1:
                    analysis['Holiday'] = {
                        'value': 'Có',
                        'impact': 'Ngày lễ → tăng doanh thu do nhu cầu mua sắm cao',
                        'direction': 'Tăng'
                    }
                else:
                    analysis['Holiday'] = {
                        'value': 'Không',
                        'impact': 'Ngày thường → doanh thu bình thường',
                        'direction': 'Trung tính'
                    }
            else:
                # Đảm bảo luôn có Holiday factor
                analysis['Holiday'] = {
                    'value': 'Không',
                    'impact': 'Ngày thường → doanh thu bình thường',
                    'direction': 'Trung tính'
                }
            
            # Đảm bảo tất cả factors đều có mặt
            required_factors = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday']
            for factor in required_factors:
                if factor not in analysis:
                    analysis[factor] = {
                        'value': 0,
                        'pct_change': 0,
                        'impact': 'Không thể phân tích',
                        'direction': 'Trung tính'
                    }
            
            return analysis
            
        except Exception as e:
            print(f"❌ Lỗi khi phân tích external factors: {e}")
            # Trả về dictionary mặc định thay vì None
            return {
                'Temperature': {'value': 0, 'pct_change': 0, 'impact': 'Lỗi phân tích', 'direction': 'Trung tính'},
                'Fuel_Price': {'value': 0, 'pct_change': 0, 'impact': 'Lỗi phân tích', 'direction': 'Trung tính'},
                'CPI': {'value': 0, 'pct_change': 0, 'impact': 'Lỗi phân tích', 'direction': 'Trung tính'},
                'Unemployment': {'value': 0, 'pct_change': 0, 'impact': 'Lỗi phân tích', 'direction': 'Trung tính'},
                'Holiday': {'value': 'Không', 'impact': 'Lỗi phân tích', 'direction': 'Trung tính'}
            }
    
    def make_prediction(self, weekly_sales_data, external_factors_current, external_factors_previous=None):
        """
        Thực hiện dự đoán hoàn chỉnh
        
        Args:
            weekly_sales_data: List 10 tuần doanh thu quá khứ
            external_factors_current: Dict các yếu tố bên ngoài tuần dự đoán
            external_factors_previous: Dict các yếu tố bên ngoài tuần trước (để tính relative changes)
        """
        try:
            print("\n🔮 ĐANG THỰC HIỆN DỰ ĐOÁN...")
            print("=" * 50)
            
            # 1. Chuẩn bị dữ liệu
            print("📊 Chuẩn bị dữ liệu...")
            features_df, historical_df = self.prepare_input_data(weekly_sales_data, external_factors_current, external_factors_previous)
            if features_df is None:
                return None
            
            # 2. Dự đoán GRU
            print("🧠 Dự đoán GRU dựa trên doanh thu các tuần trước...")
            gru_prediction = self.predict_gru_sales(weekly_sales_data)
            if gru_prediction is None:
                return None
            
            print(f"✅ GRU Prediction: ${gru_prediction:,.2f}")
            
            # 3. Dự đoán XGBoost adjustment
            print("🌳 Dự đoán điều chỉnh từ XGBoost...")
            xgb_adjustment = self.predict_xgboost_adjustment(features_df)
            if xgb_adjustment is None:
                return None
            
            print(f"✅ XGBoost Adjustment: ${xgb_adjustment:,.2f}")
            
            # 4. Tính final prediction (80% GRU + 20% XGBoost)
            final_prediction = 0.8 * gru_prediction + 0.2 * xgb_adjustment
            
            # 5. Phân tích external factors
            print("🔍 Phân tích tác động external factors...")
            factor_analysis = self.analyze_external_factors(features_df, external_factors_current)
            
            # Đảm bảo factor_analysis không bao giờ là None
            if factor_analysis is None:
                factor_analysis = {
                    'Temperature': {'value': 0, 'pct_change': 0, 'impact': 'Không thể phân tích', 'direction': 'Trung tính'},
                    'Fuel_Price': {'value': 0, 'pct_change': 0, 'impact': 'Không thể phân tích', 'direction': 'Trung tính'},
                    'CPI': {'value': 0, 'pct_change': 0, 'impact': 'Không thể phân tích', 'direction': 'Trung tính'},
                    'Unemployment': {'value': 0, 'pct_change': 0, 'impact': 'Không thể phân tích', 'direction': 'Trung tính'},
                    'Holiday': {'value': 'Không', 'impact': 'Không thể phân tích', 'direction': 'Trung tính'}
                }
            
            # 6. Tổng hợp kết quả
            result = {
                'gru_prediction': gru_prediction,
                'xgb_adjustment': xgb_adjustment,
                'final_prediction': final_prediction,
                'factor_analysis': factor_analysis,
                'features_used': list(features_df.columns),
                'historical_data': historical_df
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Lỗi khi thực hiện dự đoán: {e}")
            return None
    
    def print_prediction_result(self, result):
        """
        In kết quả dự đoán một cách đẹp mắt
        """
        if result is None:
            return
        
        print("\n" + "=" * 60)
        print("🎯 KẾT QUẢ DỰ ĐOÁN DOANH THU WALMART")
        print("=" * 60)
        
        # Thông tin dự đoán
        print(f"\n📈 DỰ ĐOÁN:")
        print(f"   • GRU Prediction:     ${result['gru_prediction']:>12,.2f}")
        print(f"   • XGBoost Adjustment: ${result['xgb_adjustment']:>12,.2f}")
        print(f"   • Final Prediction:   ${result['final_prediction']:>12,.2f}")
        
        # Phân tích factors
        print(f"\n🔍 PHÂN TÍCH TÁC ĐỘNG:")
        for factor, info in result['factor_analysis'].items():
            if factor == 'Holiday':
                print(f"   • {factor}: {info['value']}")
                print(f"     → {info['impact']}")
            else:
                print(f"   • {factor}: {info['value']:.2f} ({info['pct_change']:+.1%})")
                print(f"     → {info['impact']}")
        
        # Giải thích cộng trừ
        print(f"\n💡 GIẢI THÍCH:")
        adjustment = result['xgb_adjustment'] - result['gru_prediction']
        if adjustment > 0:
            print(f"   XGBoost điều chỉnh TĂNG doanh thu GRU: +${adjustment:,.2f}")
            print(f"   Lý do: Các yếu tố bên ngoài có lợi cho doanh thu")
        else:
            print(f"   XGBoost điều chỉnh GIẢM doanh thu GRU: {adjustment:,.2f}")
            print(f"   Lý do: Các yếu tố bên ngoài bất lợi cho doanh thu")
        
        print(f"\n📊 TỈ LỆ ENSEMBLE: 80% GRU + 20% XGBoost")
        print("=" * 60)
    
    def run_prediction_demo(self):
        """
        Chạy demo dự đoán với dữ liệu mẫu
        """
        print("\n🎯 DEMO DỰ ĐOÁN DOANH THU WALMART")
        print("=" * 50)
        
        # Dữ liệu mẫu - 10 tuần doanh thu
        sample_sales = [
            100000, 105000, 98000, 112000, 108000,
            115000, 102000, 118000, 125000, 120000
        ]
        
        print(f"📊 Dữ liệu doanh thu 10 tuần gần nhất:")
        for i, sales in enumerate(sample_sales, 1):
            print(f"   Tuần {i:2d}: ${sales:>8,}")
        
        # External factors cho tuần TRƯỚC (tuần 10)
        sample_factors_previous = {
            'Temperature': 25,      # Nhiệt độ tuần trước
            'Fuel_Price': 3.5,     # Giá xăng tuần trước
            'CPI': 250,            # CPI tuần trước
            'Unemployment': 5.0,   # Thất nghiệp tuần trước
            'Holiday_Flag': 0      # Không phải ngày lễ
        }
        
        # External factors cho tuần HIỆN TẠI (tuần 11 - cần dự đoán)
        sample_factors_current = {
            'Temperature': 28,      # Nhiệt độ tăng
            'Fuel_Price': 3.8,     # Giá xăng tăng
            'CPI': 255,            # CPI tăng
            'Unemployment': 5.2,   # Thất nghiệp tăng
            'Holiday_Flag': 0      # Không phải ngày lễ
        }
        
        print(f"\n🌍 External factors tuần TRƯỚC (tuần 10):")
        for factor, value in sample_factors_previous.items():
            if factor == 'Holiday_Flag':
                print(f"   {factor}: {'Có' if value == 1 else 'Không'}")
            else:
                print(f"   {factor}: {value}")
        
        print(f"\n🌍 External factors tuần HIỆN TẠI (tuần 11):")
        for factor, value in sample_factors_current.items():
            if factor == 'Holiday_Flag':
                print(f"   {factor}: {'Có' if value == 1 else 'Không'}")
            else:
                print(f"   {factor}: {value}")
        
        # Tính % thay đổi để demo
        print(f"\n📊 % Thay đổi so với tuần trước:")
        for factor in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
            prev_val = sample_factors_previous[factor]
            curr_val = sample_factors_current[factor]
            if prev_val != 0:
                pct_change = (curr_val - prev_val) / prev_val * 100
                print(f"   {factor}: {pct_change:+.1f}% ({prev_val} → {curr_val})")
        
        # Thực hiện dự đoán
        result = self.make_prediction(sample_sales, sample_factors_current, sample_factors_previous)
        
        # In kết quả
        self.print_prediction_result(result)
        
        return result
    
    def interactive_prediction(self):
        """
        Cho phép người dùng nhập dữ liệu tương tác
        """
        print("\n🎯 DỰ ĐOÁN TƯƠNG TÁC")
        print("=" * 50)
        
        try:
            # Nhập doanh thu 10 tuần
            print("📊 Nhập doanh thu 10 tuần gần nhất (phân cách bằng dấu phẩy):")
            print("Ví dụ: 100000, 105000, 98000, 112000, 108000, 115000, 102000, 118000, 125000, 120000")
            
            sales_input = input("👉 Doanh thu 10 tuần: ").strip()
            weekly_sales = [float(x.strip()) for x in sales_input.split(',')]
            
            if len(weekly_sales) != 10:
                print("❌ Vui lòng nhập đúng 10 tuần!")
                return None
            
            print(f"✅ Đã nhập {len(weekly_sales)} tuần doanh thu")
            
            # Nhập external factors tuần TRƯỚC
            print(f"\n🌍 Nhập external factors tuần TRƯỚC (tuần 10):")
            
            temp_prev = float(input("👉 Nhiệt độ tuần trước (°C): "))
            fuel_prev = float(input("👉 Giá xăng tuần trước ($/gallon): "))
            cpi_prev = float(input("👉 CPI tuần trước: "))
            unemp_prev = float(input("👉 Tỷ lệ thất nghiệp tuần trước (%): "))
            holiday_prev = int(input("👉 Ngày lễ tuần trước (1=Có, 0=Không): "))
            
            external_factors_previous = {
                'Temperature': temp_prev,
                'Fuel_Price': fuel_prev,
                'CPI': cpi_prev,
                'Unemployment': unemp_prev,
                'Holiday_Flag': holiday_prev
            }
            
            print("✅ Đã nhập external factors tuần trước")
            
            # Nhập external factors tuần HIỆN TẠI
            print(f"\n🌍 Nhập external factors tuần HIỆN TẠI (tuần 11 - cần dự đoán):")
            
            temp_curr = float(input("👉 Nhiệt độ tuần này (°C): "))
            fuel_curr = float(input("👉 Giá xăng tuần này ($/gallon): "))
            cpi_curr = float(input("👉 CPI tuần này: "))
            unemp_curr = float(input("👉 Tỷ lệ thất nghiệp tuần này (%): "))
            holiday_curr = int(input("👉 Ngày lễ tuần này (1=Có, 0=Không): "))
            
            external_factors_current = {
                'Temperature': temp_curr,
                'Fuel_Price': fuel_curr,
                'CPI': cpi_curr,
                'Unemployment': unemp_curr,
                'Holiday_Flag': holiday_curr
            }
            
            print("✅ Đã nhập external factors tuần hiện tại")
            
            # Hiển thị % thay đổi
            print(f"\n📊 % Thay đổi so với tuần trước:")
            for factor in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
                prev_val = external_factors_previous[factor]
                curr_val = external_factors_current[factor]
                if prev_val != 0:
                    pct_change = (curr_val - prev_val) / prev_val * 100
                    print(f"   {factor}: {pct_change:+.1f}% ({prev_val} → {curr_val})")
            
            # Thực hiện dự đoán
            result = self.make_prediction(weekly_sales, external_factors_current, external_factors_previous)
            
            # In kết quả
            self.print_prediction_result(result)
            
            return result
            
        except ValueError as e:
            print(f"❌ Lỗi nhập liệu: {e}")
            return None
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return None

# ========== FASTAPI INTEGRATION ==========
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Dict, Optional
    import uvicorn
    
    # Tạo FastAPI app
    app = FastAPI(
        title="Walmart Sales Prediction API",
        description="API dự đoán doanh thu Walmart sử dụng GRU + XGBoost Ensemble",
        version="1.0.0"
    )
    
    # Pydantic models cho API
    class WeeklySalesInput(BaseModel):
        """Input cho 10 tuần doanh thu quá khứ"""
        weekly_sales: List[float]
        description: str = "Doanh thu 10 tuần gần nhất (phân cách bằng dấu phẩy)"
    
    class ExternalFactorsInput(BaseModel):
        """Input cho external factors tuần dự đoán"""
        temperature: float
        fuel_price: float
        cpi: float
        unemployment: float
        holiday_flag: int = 0
        description: str = "External factors cho tuần dự đoán"
    
    class ExternalFactorsPreviousInput(BaseModel):
        """Input cho external factors tuần trước (để tính relative changes)"""
        temperature: float
        fuel_price: float
        cpi: float
        unemployment: float
        holiday_flag: int = 0
        description: str = "External factors tuần trước (để tính % thay đổi)"
    
    class PredictionRequest(BaseModel):
        """Request hoàn chỉnh cho dự đoán"""
        weekly_sales: List[float]
        external_factors_current: ExternalFactorsInput
        external_factors_previous: ExternalFactorsPreviousInput
    
    class PredictionResponse(BaseModel):
        """Response dự đoán"""
        gru_prediction: float
        xgb_adjustment: float
        final_prediction: float
        factor_analysis: Dict
        explanation: str
        success: bool
        message: str
    
    # Global API instance
    api_instance = None
    
    @app.on_event("startup")
    async def startup_event():
        """Khởi tạo API khi startup"""
        global api_instance
        api_instance = WalmartPredictionAPI()
        if not api_instance.is_model_loaded:
            raise RuntimeError("Không thể load model")
        print("✅ FastAPI đã khởi tạo thành công!")
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Walmart Sales Prediction API",
            "version": "1.0.0",
            "endpoints": {
                "/": "API info",
                "/health": "Health check",
                "/predict": "Dự đoán doanh thu",
                "/demo": "Demo với dữ liệu mẫu"
            }
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        global api_instance
        if api_instance and api_instance.is_model_loaded:
            return {"status": "healthy", "model_loaded": True}
        else:
            return {"status": "unhealthy", "model_loaded": False}
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict_sales(request: PredictionRequest):
        """Endpoint dự đoán doanh thu"""
        global api_instance
        
        if not api_instance or not api_instance.is_model_loaded:
            raise HTTPException(status_code=500, detail="Model chưa được load")
        
        try:
            # Validate input
            if len(request.weekly_sales) != 10:
                raise HTTPException(status_code=400, detail="Cần đúng 10 tuần doanh thu")
            
            # Chuẩn bị external factors
            external_factors_current = {
                'Temperature': request.external_factors_current.temperature,
                'Fuel_Price': request.external_factors_current.fuel_price,
                'CPI': request.external_factors_current.cpi,
                'Unemployment': request.external_factors_current.unemployment,
                'Holiday_Flag': request.external_factors_current.holiday_flag
            }
            
            # Tạo một dict để truyền cho prepare_input_data
            # Để tính relative changes, chúng ta cần dữ liệu tuần trước.
            # Vì là demo, chúng ta sẽ sử dụng giá trị mặc định cho tuần trước.
            # Trong thực tế, bạn sẽ cần lấy dữ liệu tuần trước từ cơ sở dữ liệu hoặc API khác.
            external_factors_previous = {
                'Temperature': request.external_factors_previous.temperature,
                'Fuel_Price': request.external_factors_previous.fuel_price,
                'CPI': request.external_factors_previous.cpi,
                'Unemployment': request.external_factors_previous.unemployment,
                'Holiday_Flag': request.external_factors_previous.holiday_flag
            }
            
            # Thực hiện dự đoán
            result = api_instance.make_prediction(request.weekly_sales, external_factors_current, external_factors_previous)
            
            if result is None:
                raise HTTPException(status_code=500, detail="Lỗi khi thực hiện dự đoán")
            
            # Đảm bảo factor_analysis luôn hợp lệ
            if result.get('factor_analysis') is None:
                result['factor_analysis'] = {
                    'Temperature': {'value': 0, 'pct_change': 0, 'impact': 'Không thể phân tích', 'direction': 'Trung tính'},
                    'Fuel_Price': {'value': 0, 'pct_change': 0, 'impact': 'Không thể phân tích', 'direction': 'Trung tính'},
                    'CPI': {'value': 0, 'pct_change': 0, 'impact': 'Không thể phân tích', 'direction': 'Trung tính'},
                    'Unemployment': {'value': 0, 'pct_change': 0, 'impact': 'Không thể phân tích', 'direction': 'Trung tính'},
                    'Holiday': {'value': 'Không', 'impact': 'Không thể phân tích', 'direction': 'Trung tính'}
                }
            
            # Tạo explanation
            adjustment = result['xgb_adjustment'] - result['gru_prediction']
            if adjustment > 0:
                explanation = f"XGBoost điều chỉnh TĂNG doanh thu GRU: +${adjustment:,.2f}. Lý do: Các yếu tố bên ngoài có lợi cho doanh thu."
            else:
                explanation = f"XGBoost điều chỉnh GIẢM doanh thu GRU: {adjustment:,.2f}. Lý do: Các yếu tố bên ngoài bất lợi cho doanh thu."
            
            return PredictionResponse(
                gru_prediction=result['gru_prediction'],
                xgb_adjustment=result['xgb_adjustment'],
                final_prediction=result['final_prediction'],
                factor_analysis=result['factor_analysis'],
                explanation=explanation,
                success=True,
                message="Dự đoán thành công"
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")
    
    @app.get("/demo")
    async def demo_prediction():
        """Demo endpoint với dữ liệu mẫu"""
        global api_instance
        
        if not api_instance or not api_instance.is_model_loaded:
            raise HTTPException(status_code=500, detail="Model chưa được load")
        
        try:
            # Dữ liệu mẫu
            sample_sales = [100000, 105000, 98000, 112000, 108000, 115000, 102000, 118000, 125000, 120000]
            sample_factors_current = {
                'Temperature': 28,
                'Fuel_Price': 3.8,
                'CPI': 255,
                'Unemployment': 5.2,
                'Holiday_Flag': 0
            }
            sample_factors_previous = {
                'Temperature': 25,
                'Fuel_Price': 3.5,
                'CPI': 250,
                'Unemployment': 5.0,
                'Holiday_Flag': 0
            }
            
            # Thực hiện dự đoán
            result = api_instance.make_prediction(sample_sales, sample_factors_current, sample_factors_previous)
            
            if result is None:
                raise HTTPException(status_code=500, detail="Lỗi khi thực hiện demo")
            
            # Đảm bảo factor_analysis luôn hợp lệ
            if result.get('factor_analysis') is None:
                result['factor_analysis'] = {
                    'Temperature': {'value': 0, 'pct_change': 0, 'impact': 'Không thể phân tích', 'direction': 'Trung tính'},
                    'Fuel_Price': {'value': 0, 'pct_change': 0, 'impact': 'Không thể phân tích', 'direction': 'Trung tính'},
                    'CPI': {'value': 0, 'pct_change': 0, 'impact': 'Không thể phân tích', 'direction': 'Trung tính'},
                    'Unemployment': {'value': 0, 'pct_change': 0, 'impact': 'Không thể phân tích', 'direction': 'Trung tính'},
                    'Holiday': {'value': 'Không', 'impact': 'Không thể phân tích', 'direction': 'Trung tính'}
                }
            
            return {
                "success": True,
                "message": "Demo thành công",
                "sample_data": {
                    "weekly_sales": sample_sales,
                    "external_factors_current": sample_factors_current,
                    "external_factors_previous": sample_factors_previous
                },
                "relative_changes": {
                    "Temperature": "+12.0% (25°C → 28°C)",
                    "Fuel_Price": "+8.6% ($3.50 → $3.80)",
                    "CPI": "+2.0% (250 → 255)",
                    "Unemployment": "+4.0% (5.0% → 5.2%)"
                },
                "prediction_result": {
                    "gru_prediction": result['gru_prediction'],
                    "xgb_adjustment": result['xgb_adjustment'],
                    "final_prediction": result['final_prediction'],
                    "factor_analysis": result['factor_analysis']
                }
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi demo: {str(e)}")
    
    @app.get("/model-info")
    async def get_model_info():
        """Lấy thông tin model"""
        global api_instance
        
        if not api_instance or not api_instance.model_info is None:
            return {"model_info": "Không có thông tin model"}
        
        return {"model_info": api_instance.model_info}
    
    def run_fastapi():
        """Chạy FastAPI server"""
        print("🚀 Khởi động FastAPI server...")
        print("📱 API sẽ chạy tại: http://localhost:8000")
        print("📚 API docs: http://localhost:8000/docs")
        print("🔍 Health check: http://localhost:8000/health")
        print("🎯 Dự đoán: POST http://localhost:8000/predict")
        print("🎮 Demo: GET http://localhost:8000/demo")
        print("\n💡 Để deploy lên host:")
        print("   1. Chạy: uvicorn walmart_prediction_api:app --host 0.0.0.0 --port 8000")
        print("   2. Hoặc: python walmart_prediction_api.py --web")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # Thêm option chạy FastAPI vào main menu
    def main_with_fastapi():
        """Main function với option FastAPI"""
        print("🏪 WALMART SALES PREDICTION API")
        print("=" * 50)
        print("🎯 Model: Relative Changes Approach")
        print("📊 Dự đoán điều chỉnh doanh thu dựa trên % thay đổi của external factors")
        print("🔄 Cần dữ liệu: Tuần trước + Tuần hiện tại để tính % thay đổi")
        print("=" * 50)
        
        # Khởi tạo API
        api = WalmartPredictionAPI()
        
        if not api.is_model_loaded:
            print("❌ Không thể khởi tạo API. Vui lòng kiểm tra model.")
            return
        
        while True:
            print("\n📋 MENU LỰA CHỌN:")
            print("1. 🎯 Demo dự đoán (dữ liệu mẫu)")
            print("2. 📊 Dự đoán tương tác (nhập dữ liệu)")
            print("3. 📋 Thông tin model")
            print("4. 🌐 Chạy FastAPI Web Server")
            print("5. 🚪 Thoát")
            print("\n💡 Lưu ý: Model Relative Changes cần dữ liệu tuần trước để tính % thay đổi")
            
            choice = input("\n👉 Nhập lựa chọn (1-5): ").strip()
            
            if choice == '1':
                api.run_prediction_demo()
            elif choice == '2':
                api.interactive_prediction()
            elif choice == '3':
                if api.model_info:
                    print("\n📋 THÔNG TIN MODEL:")
                    print("=" * 50)
                    print(api.model_info)
                else:
                    print("❌ Không có thông tin model")
            elif choice == '4':
                print("🌐 Khởi động FastAPI Web Server...")
                run_fastapi()
                break
            elif choice == '5':
                print("👋 Tạm biệt!")
                break
            else:
                print("❌ Lựa chọn không hợp lệ. Vui lòng chọn 1-5.")
    
    # Thay thế main function cũ
    if __name__ == "__main__":
        import sys
        
        # Kiểm tra argument để chạy FastAPI
        if len(sys.argv) > 1 and sys.argv[1] == "--web":
            # Chạy FastAPI trực tiếp
            api = WalmartPredictionAPI()
            if api.is_model_loaded:
                run_fastapi()
            else:
                print("❌ Không thể load model để chạy FastAPI")
        else:
            # Chạy CLI với option FastAPI
            main_with_fastapi()

except ImportError as e:
    print(f"⚠️ FastAPI không khả dụng: {e}")
    print("💡 Để sử dụng FastAPI, cài đặt: pip install fastapi uvicorn")
    print("🔧 API vẫn hoạt động ở chế độ CLI")
    
    if __name__ == "__main__":
        main_with_fastapi()
