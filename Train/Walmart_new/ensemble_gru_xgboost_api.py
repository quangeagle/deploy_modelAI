# Ensemble GRU + XGBoost Sales Prediction API
# API kết hợp GRU prediction + XGBoost adjustment

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pickle
import warnings
import joblib
from datetime import datetime
import shap
warnings.filterwarnings('ignore')

# ========== 1. IMPROVED GRU MODEL CLASS ==========
class ImprovedGRUSalesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(ImprovedGRUSalesPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers với attention
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout
        )
        
        # Additional layers
        self.fc1 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Activation functions
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # GRU processing
        gru_out, _ = self.gru(x)
        
        # Apply layer normalization
        gru_out = self.layer_norm(gru_out)
        
        # Self-attention mechanism
        gru_out = gru_out.transpose(0, 1)
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        attn_out = attn_out.transpose(0, 1)
        
        # Global average pooling + max pooling
        avg_pool = torch.mean(attn_out, dim=1)
        max_pool, _ = torch.max(attn_out, dim=1)
        
        # Combine pooling results
        combined = torch.cat([avg_pool, max_pool], dim=1)
        
        # Fully connected layers
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out

# ========== 2. INPUT/OUTPUT MODELS ==========
class EnsemblePredictionRequest(BaseModel):
    """Input model cho Ensemble API"""
    sales_history: List[float]  # List 10 giá trị doanh thu tuần trước
    external_factors_current: Optional[dict] = None  # External factors tuần hiện tại
    external_factors_previous: Optional[dict] = None  # External factors tuần trước (để tính change)

class EnsemblePredictionResponse(BaseModel):
    """Output model cho Ensemble API"""
    gru_prediction: float
    xgboost_adjustment: float
    final_prediction: float
    confidence_score: float
    input_sequence: List[float]
    external_factors_current: dict
    external_factors_previous: dict
    message: str
    ensemble_breakdown: dict
    adjustment_explanation: dict  # Giải thích chi tiết adjustment với SHAP

# ========== 3. MODEL LOADER ==========
class EnsembleModelLoader:
    def __init__(self):
        self.gru_model = None
        self.xgb_model = None
        self.sequence_scaler = None
        self.target_scaler = None
        self.feature_columns = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_models()
    
    def load_models(self):
        """Load cả GRU và XGBoost models"""
        try:
            print("🔄 Loading Ensemble Models...")
            
            # 1. Load GRU Model
            print("📊 Loading Improved GRU model...")
            self.gru_model = ImprovedGRUSalesPredictor(input_size=1, hidden_size=128, num_layers=2, dropout=0.2)
            self.gru_model.load_state_dict(torch.load('improved_gru_model.pth', map_location=self.device))
            self.gru_model.to(self.device)
            self.gru_model.eval()
            
            # Load GRU scalers
            with open('improved_sequence_scaler.pkl', 'rb') as f:
                self.sequence_scaler = pickle.load(f)
            
            with open('improved_target_scaler.pkl', 'rb') as f:
                self.target_scaler = pickle.load(f)
            
            # 2. Load XGBoost Model
            print("🌳 Loading XGBoost Adjustment model...")
            self.xgb_model = joblib.load('output_relative/xgb_model.pkl')
            
            # Load feature columns
            with open('output_relative/feature_columns.txt', 'r') as f:
                self.feature_columns = f.read().splitlines()
            
            print("✅ Ensemble models loaded successfully!")
            print(f"🔧 Device: {self.device}")
            print(f"🌳 XGBoost features: {len(self.feature_columns)} features")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise e

# ========== 4. FEATURE PREPARATION ==========
def create_business_reason(feature_name: str, feature_value: float, shap_value: float, external_factors_current: dict, external_factors_previous: dict) -> str:
    """
    Tạo business reason dựa trên feature name và SHAP value
    """
    try:
        # Temperature change
        if 'Temperature_change' in feature_name:
            if shap_value > 0:
                return "Nhiệt độ tăng → Khách hàng thoải mái mua sắm → Tăng doanh thu"
            else:
                return "Nhiệt độ giảm → Khách hàng ít ra ngoài → Giảm doanh thu"
        
        # Fuel Price change
        elif 'Fuel_Price_change' in feature_name:
            if shap_value > 0:
                return "Giá xăng giảm → Chi phí vận chuyển thấp → Tăng doanh thu"
            else:
                return "Giá xăng tăng → Chi phí vận chuyển cao → Giảm doanh thu"
        
        # CPI change
        elif 'CPI_change' in feature_name:
            if shap_value > 0:
                return "CPI giảm → Giảm phát → Sức mua tăng → Tăng doanh thu"
            else:
                return "CPI tăng → Lạm phát cao → Sức mua giảm → Giảm doanh thu"
        
        # Unemployment change
        elif 'Unemployment_change' in feature_name:
            if shap_value > 0:
                return "Thất nghiệp giảm → Sức mua tăng → Tăng doanh thu"
            else:
                return "Thất nghiệp tăng → Sức mua giảm → Giảm doanh thu"
        
        # Holiday Flag
        elif 'Holiday_Flag' in feature_name:
            if feature_value == 1:
                return "Có ngày lễ → Khách hàng mua sắm nhiều → Tăng doanh thu"
            else:
                return "Không có ngày lễ → Doanh thu bình thường"
        
        # Month/Season
        elif 'Month' in feature_name:
            month = int(feature_value)
            if month in [11, 12]:
                return f"Tháng {month} - Mùa mua sắm cuối năm → Doanh thu cao"
            elif month in [1, 2]:
                return f"Tháng {month} - Sau lễ, mùa thấp điểm → Doanh thu thấp"
            else:
                return f"Tháng {month} - Mùa bình thường → Doanh thu ổn định"
        
        # Week of Year
        elif 'WeekOfYear' in feature_name:
            week = int(feature_value)
            if week in [50, 51, 52]:
                return f"Tuần {week} - Gần Giáng sinh → Doanh thu cao"
            elif week in [1, 2, 3]:
                return f"Tuần {week} - Sau lễ → Doanh thu thấp"
            else:
                return f"Tuần {week} - Tuần bình thường"
        
        # Weekend
        elif 'Is_Weekend' in feature_name:
            if feature_value == 1:
                return "Cuối tuần → Khách hàng có thời gian mua sắm → Tăng doanh thu"
            else:
                return "Ngày thường → Doanh thu bình thường"
        
        # Default
        else:
            if shap_value > 0:
                return f"{feature_name} có ảnh hưởng tích cực đến doanh thu"
            elif shap_value < 0:
                return f"{feature_name} có ảnh hưởng tiêu cực đến doanh thu"
            else:
                return f"{feature_name} không ảnh hưởng đáng kể"
                
    except Exception as e:
        return f"Không thể tạo business reason cho {feature_name}"

def create_adjustment_explanation_with_shap(model_loader, features_array: np.ndarray, external_factors_current: dict, external_factors_previous: dict, xgb_adjustment: float) -> dict:
    """
    Tạo giải thích chi tiết về XGBoost adjustment sử dụng SHAP values thực sự
    """
    try:
        explanation = {
            "summary": "",
            "factor_analysis": {},
            "overall_impact": "",
            "business_logic": [],
            "shap_analysis": {},
            "contribution_breakdown": {}
        }
        
        # ========== STEP 1: TÍNH SHAP VALUES ==========
        print("🔍 Calculating SHAP values...")
        
        # Tạo TreeExplainer cho XGBoost
        explainer = shap.TreeExplainer(model_loader.xgb_model)
        
        # Tính SHAP values
        shap_values = explainer.shap_values(features_array)
        
        # Lấy feature names
        feature_names = model_loader.feature_columns
        
        # Tạo DataFrame để dễ xử lý
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values[0],  # Lấy SHAP values cho sample đầu tiên
            'feature_value': features_array[0]  # Lấy feature values
        })
        
        # Sắp xếp theo absolute SHAP values
        shap_df['abs_shap'] = abs(shap_df['shap_value'])
        shap_df = shap_df.sort_values('abs_shap', ascending=False)
        
        print(f"✅ SHAP values calculated: {len(shap_df)} features")
        
        # ========== STEP 2: PHÂN TÍCH TỪNG FACTOR ==========
        factor_impacts = {}
        total_shap = abs(shap_df['shap_value'].sum())
        
        for _, row in shap_df.iterrows():
            feature_name = row['feature']
            shap_value = row['shap_value']
            feature_value = row['feature_value']
            
            # Tính contribution percentage
            contribution_pct = (abs(shap_value) / total_shap * 100) if total_shap > 0 else 0
            
            # Xác định impact direction
            if shap_value > 0:
                impact = "positive"
                direction = "tăng"
            elif shap_value < 0:
                impact = "negative"
                direction = "giảm"
            else:
                impact = "neutral"
                direction = "không ảnh hưởng"
            
            # Tạo business reason dựa trên feature type
            business_reason = create_business_reason(feature_name, feature_value, shap_value, external_factors_current, external_factors_previous)
            
            factor_impacts[feature_name] = {
                "feature_value": feature_value,
                "shap_value": shap_value,
                "impact": impact,
                "direction": direction,
                "contribution_percentage": f"{contribution_pct:.1f}%",
                "business_reason": business_reason
            }
        
        # ========== STEP 3: TẠO SHAP ANALYSIS ==========
        shap_analysis = {
            "total_positive_contribution": float(shap_df[shap_df['shap_value'] > 0]['shap_value'].sum()),
            "total_negative_contribution": float(shap_df[shap_df['shap_value'] < 0]['shap_value'].sum()),
            "net_contribution": float(shap_df['shap_value'].sum()),
            "top_positive_features": shap_df[shap_df['shap_value'] > 0].head(3)[['feature', 'shap_value']].to_dict('records'),
            "top_negative_features": shap_df[shap_df['shap_value'] < 0].head(3)[['feature', 'shap_value']].to_dict('records'),
            "feature_ranking": shap_df[['feature', 'shap_value', 'abs_shap']].to_dict('records')
        }
        
        # ========== STEP 4: TẠO CONTRIBUTION BREAKDOWN ==========
        contribution_breakdown = {}
        for _, row in shap_df.iterrows():
            feature_name = row['feature']
            shap_value = row['shap_value']
            contribution_pct = (abs(shap_value) / total_shap * 100) if total_shap > 0 else 0
            
            if abs(shap_value) > 0.01:  # Chỉ hiển thị features có ảnh hưởng đáng kể
                contribution_breakdown[feature_name] = {
                    "shap_value": float(shap_value),
                    "contribution_percentage": f"{contribution_pct:.1f}%",
                    "direction": "tăng" if shap_value > 0 else "giảm",
                    "interpretation": f"Đóng góp {contribution_pct:.1f}% vào adjustment ({'tăng' if shap_value > 0 else 'giảm'} {abs(shap_value):,.0f})"
                }
        
        # ========== STEP 5: TẠO BUSINESS LOGIC ==========
        business_logic = []
        if xgb_adjustment > 0:
            business_logic.append(f"XGBoost dự đoán adjustment dương (+{xgb_adjustment:,.0f})")
            
            # Tìm top positive contributors
            top_positive = shap_df[shap_df['shap_value'] > 0].head(3)
            if not top_positive.empty:
                top_features = top_positive['feature'].tolist()
                business_logic.append(f"Chủ yếu do: {', '.join(top_features)}")
            
            # Tìm top negative contributors (nếu có)
            top_negative = shap_df[shap_df['shap_value'] < 0].head(3)
            if not top_negative.empty:
                top_features = top_negative['feature'].tolist()
                business_logic.append(f"Mặc dù có: {', '.join(top_features)}")
        else:
            business_logic.append(f"XGBoost dự đoán adjustment âm ({xgb_adjustment:,.0f})")
            
            # Tìm top negative contributors
            top_negative = shap_df[shap_df['shap_value'] < 0].head(3)
            if not top_negative.empty:
                top_features = top_negative['feature'].tolist()
                business_logic.append(f"Chủ yếu do: {', '.join(top_features)}")
            
            # Tìm top positive contributors (nếu có)
            top_positive = shap_df[shap_df['shap_value'] > 0].head(3)
            if not top_positive.empty:
                top_features = top_positive['feature'].tolist()
                business_logic.append(f"Mặc dù có: {', '.join(top_features)}")
        
        # ========== STEP 6: TẠO OVERALL IMPACT ==========
        positive_count = len(shap_df[shap_df['shap_value'] > 0])
        negative_count = len(shap_df[shap_df['shap_value'] < 0])
        
        if positive_count > negative_count:
            overall_impact = "Tổng thể thuận lợi cho doanh thu"
        elif negative_count > positive_count:
            overall_impact = "Tổng thể bất lợi cho doanh thu"
        else:
            overall_impact = "Tổng thể cân bằng"
        
        # ========== STEP 7: TẠO SUMMARY ==========
        summary = f"XGBoost adjustment: {xgb_adjustment:,.0f} ({'tăng' if xgb_adjustment > 0 else 'giảm'} doanh thu)"
        summary += f" | SHAP: +{shap_analysis['total_positive_contribution']:,.0f} / -{abs(shap_analysis['total_negative_contribution']):,.0f}"
        
        explanation = {
            "summary": summary,
            "factor_analysis": factor_impacts,
            "overall_impact": overall_impact,
            "business_logic": business_logic,
            "shap_analysis": shap_analysis,
            "contribution_breakdown": contribution_breakdown
        }
        
        return explanation
        
    except Exception as e:
        print(f"❌ Error creating SHAP explanation: {e}")
        return {
            "summary": "Không thể tạo SHAP explanation",
            "factor_analysis": {},
            "overall_impact": "Unknown",
            "business_logic": ["Error occurred"],
            "shap_analysis": {},
            "contribution_breakdown": {}
        }

def create_adjustment_explanation(external_factors_current: dict, external_factors_previous: dict, xgb_adjustment: float) -> dict:
    """
    Tạo giải thích chi tiết về XGBoost adjustment với contribution score
    """
    try:
        explanation = {
            "summary": "",
            "factor_analysis": {},
            "overall_impact": "",
            "business_logic": [],
            "contribution_breakdown": {}
        }
        
        # Tính toán thay đổi và phân tích từng factor
        factor_impacts = {}
        contribution_scores = {}
        
        # 1. Temperature Analysis
        if 'Temperature' in external_factors_previous and 'Temperature' in external_factors_current:
            temp_prev = external_factors_previous['Temperature']
            temp_curr = external_factors_current['Temperature']
            temp_change = ((temp_curr - temp_prev) / temp_prev * 100) if temp_prev != 0 else 0
            
            # Tính contribution score cho Temperature
            if temp_change > 5:
                temp_contribution = min(25, abs(temp_change) * 0.5)  # Max 25% contribution
                factor_impacts['Temperature'] = {
                    "change": f"+{temp_change:.1f}%",
                    "impact": "positive",
                    "reason": "Nhiệt độ tăng → Khách hàng thoải mái mua sắm → Tăng doanh thu",
                    "contribution_score": f"+{temp_contribution:.1f}%"
                }
                contribution_scores['Temperature'] = temp_contribution
            elif temp_change < -5:
                temp_contribution = -min(25, abs(temp_change) * 0.5)
                factor_impacts['Temperature'] = {
                    "change": f"{temp_change:.1f}%",
                    "impact": "negative", 
                    "reason": "Nhiệt độ giảm → Khách hàng ít ra ngoài → Giảm doanh thu",
                    "contribution_score": f"{temp_contribution:.1f}%"
                }
                contribution_scores['Temperature'] = temp_contribution
            else:
                factor_impacts['Temperature'] = {
                    "change": f"{temp_change:.1f}%",
                    "impact": "neutral",
                    "reason": "Nhiệt độ ổn định → Không ảnh hưởng đáng kể",
                    "contribution_score": "0%"
                }
                contribution_scores['Temperature'] = 0
        
        # 2. Fuel Price Analysis
        if 'Fuel_Price' in external_factors_previous and 'Fuel_Price' in external_factors_current:
            fuel_prev = external_factors_previous['Fuel_Price']
            fuel_curr = external_factors_current['Fuel_Price']
            fuel_change = ((fuel_curr - fuel_prev) / fuel_prev * 100) if fuel_prev != 0 else 0
            
            if fuel_change > 10:
                fuel_contribution = -min(30, abs(fuel_change) * 0.8)  # Max 30% negative contribution
                factor_impacts['Fuel_Price'] = {
                    "change": f"+{fuel_change:.1f}%",
                    "impact": "negative",
                    "reason": "Giá xăng tăng mạnh → Chi phí vận chuyển cao → Giảm doanh thu",
                    "contribution_score": f"{fuel_contribution:.1f}%"
                }
                contribution_scores['Fuel_Price'] = fuel_contribution
            elif fuel_change < -5:
                fuel_contribution = min(20, abs(fuel_change) * 0.6)
                factor_impacts['Fuel_Price'] = {
                    "change": f"{fuel_change:.1f}%",
                    "impact": "positive",
                    "reason": "Giá xăng giảm → Chi phí vận chuyển thấp → Tăng doanh thu",
                    "contribution_score": f"+{fuel_contribution:.1f}%"
                }
                contribution_scores['Fuel_Price'] = fuel_contribution
            else:
                factor_impacts['Fuel_Price'] = {
                    "change": f"{fuel_change:.1f}%",
                    "impact": "neutral",
                    "reason": "Giá xăng ổn định → Không ảnh hưởng đáng kể",
                    "contribution_score": "0%"
                }
                contribution_scores['Fuel_Price'] = 0
        
        # 3. CPI Analysis
        if 'CPI' in external_factors_previous and 'CPI' in external_factors_current:
            cpi_prev = external_factors_previous['CPI']
            cpi_curr = external_factors_current['CPI']
            cpi_change = ((cpi_curr - cpi_prev) / cpi_prev * 100) if cpi_prev != 0 else 0
            
            if cpi_change > 3:
                cpi_contribution = -min(25, abs(cpi_change) * 2.0)  # CPI có ảnh hưởng mạnh
                factor_impacts['CPI'] = {
                    "change": f"+{cpi_change:.1f}%",
                    "impact": "negative",
                    "reason": "CPI tăng → Lạm phát cao → Sức mua giảm → Giảm doanh thu",
                    "contribution_score": f"{cpi_contribution:.1f}%"
                }
                contribution_scores['CPI'] = cpi_contribution
            elif cpi_change < -2:
                cpi_contribution = min(20, abs(cpi_change) * 1.5)
                factor_impacts['CPI'] = {
                    "change": f"{cpi_change:.1f}%",
                    "impact": "positive",
                    "reason": "CPI giảm → Giảm phát → Sức mua tăng → Tăng doanh thu",
                    "contribution_score": f"+{cpi_contribution:.1f}%"
                }
                contribution_scores['CPI'] = cpi_contribution
            else:
                factor_impacts['CPI'] = {
                    "change": f"{cpi_change:.1f}%",
                    "impact": "neutral",
                    "reason": "CPI ổn định → Không ảnh hưởng đáng kể",
                    "contribution_score": "0%"
                }
                contribution_scores['CPI'] = 0
        
        # 4. Unemployment Analysis
        if 'Unemployment' in external_factors_previous and 'Unemployment' in external_factors_current:
            unemp_prev = external_factors_previous['Unemployment']
            unemp_curr = external_factors_current['Unemployment']
            unemp_change = ((unemp_curr - unemp_prev) / unemp_prev * 100) if unemp_prev != 0 else 0
            
            if unemp_change > 20:
                unemp_contribution = -min(35, abs(unemp_change) * 0.7)  # Unemployment có ảnh hưởng rất mạnh
                factor_impacts['Unemployment'] = {
                    "change": f"+{unemp_change:.1f}%",
                    "impact": "negative",
                    "reason": "Thất nghiệp tăng mạnh → Sức mua giảm → Giảm doanh thu",
                    "contribution_score": f"{unemp_contribution:.1f}%"
                }
                contribution_scores['Unemployment'] = unemp_contribution
            elif unemp_change < -10:
                unemp_contribution = min(30, abs(unemp_change) * 0.8)
                factor_impacts['Unemployment'] = {
                    "change": f"{unemp_change:.1f}%",
                    "impact": "positive",
                    "reason": "Thất nghiệp giảm → Sức mua tăng → Tăng doanh thu",
                    "contribution_score": f"+{unemp_contribution:.1f}%"
                }
                contribution_scores['Unemployment'] = unemp_contribution
            else:
                factor_impacts['Unemployment'] = {
                    "change": f"{unemp_change:.1f}%",
                    "impact": "neutral",
                    "reason": "Thất nghiệp ổn định → Không ảnh hưởng đáng kể",
                    "contribution_score": "0%"
                }
                contribution_scores['Unemployment'] = 0
        
        # 5. Holiday Analysis
        if 'Holiday_Flag' in external_factors_current:
            holiday_flag = external_factors_current['Holiday_Flag']
            if holiday_flag == 1:
                holiday_contribution = 15  # Holiday có ảnh hưởng cố định
                factor_impacts['Holiday_Flag'] = {
                    "change": "Có ngày lễ",
                    "impact": "positive",
                    "reason": "Ngày lễ → Khách hàng mua sắm nhiều → Tăng doanh thu",
                    "contribution_score": f"+{holiday_contribution:.1f}%"
                }
                contribution_scores['Holiday_Flag'] = holiday_contribution
            else:
                factor_impacts['Holiday_Flag'] = {
                    "change": "Không có lễ",
                    "impact": "neutral",
                    "reason": "Không có ngày lễ → Doanh thu bình thường",
                    "contribution_score": "0%"
                }
                contribution_scores['Holiday_Flag'] = 0
        
        # 6. Seasonal Analysis
        if 'Month' in external_factors_current:
            month = external_factors_current['Month']
            if month in [11, 12]:  # Tháng 11-12
                season_contribution = 20  # Mùa mua sắm cuối năm
                factor_impacts['Season'] = {
                    "change": f"Tháng {month}",
                    "impact": "positive",
                    "reason": "Mùa mua sắm cuối năm → Doanh thu cao",
                    "contribution_score": f"+{season_contribution:.1f}%"
                }
                contribution_scores['Season'] = season_contribution
            elif month in [1, 2]:  # Tháng 1-2
                season_contribution = -15  # Sau lễ, mùa thấp điểm
                factor_impacts['Season'] = {
                    "change": f"Tháng {month}",
                    "impact": "negative",
                    "reason": "Sau lễ, mùa thấp điểm → Doanh thu thấp",
                    "contribution_score": f"{season_contribution:.1f}%"
                }
                contribution_scores['Season'] = season_contribution
            else:
                factor_impacts['Season'] = {
                    "change": f"Tháng {month}",
                    "impact": "neutral",
                    "reason": "Mùa bình thường → Doanh thu ổn định",
                    "contribution_score": "0%"
                }
                contribution_scores['Season'] = 0
        
        # Tính tổng contribution và normalize
        total_contribution = sum(contribution_scores.values())
        if total_contribution != 0:
            # Normalize để tổng = 100% (hoặc -100% nếu âm)
            normalization_factor = 100 / abs(total_contribution)
            normalized_contributions = {k: v * normalization_factor for k, v in contribution_scores.items()}
        else:
            normalized_contributions = contribution_scores
        
        # Tạo contribution breakdown
        contribution_breakdown = {}
        for factor, score in normalized_contributions.items():
            if score != 0:
                contribution_breakdown[factor] = {
                    "raw_score": f"{score:.1f}%",
                    "interpretation": f"Đóng góp {abs(score):.1f}% vào adjustment",
                    "direction": "tăng" if score > 0 else "giảm"
                }
        
        # Tạo summary và overall impact
        positive_factors = [k for k, v in factor_impacts.items() if v['impact'] == 'positive']
        negative_factors = [k for k, v in factor_impacts.items() if v['impact'] == 'negative']
        neutral_factors = [k for k, v in factor_impacts.items() if v['impact'] == 'neutral']
        
        # Tạo business logic
        business_logic = []
        if xgb_adjustment > 0:
            business_logic.append(f"XGBoost dự đoán adjustment dương (+{xgb_adjustment:,.0f})")
            if positive_factors:
                business_logic.append(f"Chủ yếu do: {', '.join(positive_factors)}")
            if negative_factors:
                business_logic.append(f"Mặc dù có: {', '.join(negative_factors)}")
        else:
            business_logic.append(f"XGBoost dự đoán adjustment âm ({xgb_adjustment:,.0f})")
            if negative_factors:
                business_logic.append(f"Chủ yếu do: {', '.join(negative_factors)}")
            if positive_factors:
                business_logic.append(f"Mặc dù có: {', '.join(positive_factors)}")
        
        # Tạo overall impact
        if len(positive_factors) > len(negative_factors):
            overall_impact = "Tổng thể thuận lợi cho doanh thu"
        elif len(negative_factors) > len(positive_factors):
            overall_impact = "Tổng thể bất lợi cho doanh thu"
        else:
            overall_impact = "Tổng thể cân bằng"
        
        explanation = {
            "summary": f"XGBoost adjustment: {xgb_adjustment:,.0f} ({'tăng' if xgb_adjustment > 0 else 'giảm'} doanh thu)",
            "factor_analysis": factor_impacts,
            "overall_impact": overall_impact,
            "business_logic": business_logic,
            "contribution_breakdown": contribution_breakdown
        }
        
        return explanation
        
    except Exception as e:
        print(f"❌ Error creating adjustment explanation: {e}")
        return {
            "summary": "Không thể tạo giải thích",
            "factor_analysis": {},
            "overall_impact": "Unknown",
            "business_logic": ["Error occurred"]
        }

def prepare_external_features_with_changes(external_factors_current: dict, external_factors_previous: dict, feature_columns: List[str]) -> np.ndarray:
    """
    Chuẩn bị external features cho XGBoost với tính toán thay đổi tương đối
    """
    try:
        # Tạo DataFrame với external factors hiện tại
        features_df = pd.DataFrame([external_factors_current])
        
        # Tính toán thay đổi tương đối nếu có dữ liệu tuần trước
        if external_factors_previous:
            for col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
                if col in external_factors_previous and col in external_factors_current:
                    prev_val = external_factors_previous[col]
                    curr_val = external_factors_current[col]
                    
                    if prev_val != 0:  # Tránh chia cho 0
                        change_pct = ((curr_val - prev_val) / prev_val) * 100
                        features_df[f'{col}_change'] = change_pct
                    else:
                        features_df[f'{col}_change'] = 0.0
                else:
                    features_df[f'{col}_change'] = 0.0
        else:
            # Nếu không có dữ liệu tuần trước, set change = 0
            for col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
                features_df[f'{col}_change'] = 0.0
        
        # Đảm bảo tất cả required features có mặt
        for col in feature_columns:
            if col not in features_df.columns:
                # Set default values cho missing features
                if 'change' in col:
                    features_df[col] = 0.0  # No change
                elif col in ['Holiday_Flag', 'Is_Weekend']:
                    features_df[col] = 0  # Not holiday/weekend
                elif col in ['Month', 'WeekOfYear', 'Year', 'DayOfWeek']:
                    # Set current date values
                    now = datetime.now()
                    if col == 'Month':
                        features_df[col] = now.month
                    elif col == 'WeekOfYear':
                        features_df[col] = now.isocalendar()[1]
                    elif col == 'Year':
                        features_df[col] = now.year
                    elif col == 'DayOfWeek':
                        features_df[col] = now.weekday()
                else:
                    features_df[col] = 0.0
        
        # Reorder columns theo thứ tự feature_columns
        features_df = features_df[feature_columns]
        
        # Convert to numpy array
        features_array = features_df.values.astype(np.float32)
        
        print(f"✅ Prepared features with changes: {features_array.shape}")
        print(f"📊 Features: {list(features_df.columns)}")
        return features_array
        
    except Exception as e:
        print(f"❌ Error preparing features: {e}")
        raise e

# ========== 5. ENSEMBLE PREDICTION FUNCTION ==========
def predict_sales_ensemble(model_loader, sales_history: List[float], external_factors_current: Optional[dict] = None, external_factors_previous: Optional[dict] = None) -> dict:
    """
    Dự đoán doanh thu với ensemble GRU + XGBoost
    """
    try:
        # Validate input
        if len(sales_history) != 10:
            raise ValueError("Cần đúng 10 giá trị doanh thu tuần trước")
        
        print("🔄 Starting ensemble prediction...")
        
        # ========== STEP 1: GRU PREDICTION ==========
        print("📊 Step 1: GRU prediction...")
        
        # Convert to numpy array
        sequence = np.array(sales_history, dtype=np.float32)
        
        # Reshape to (1, 10, 1) for batch processing
        sequence = sequence.reshape(1, -1, 1)
        
        # Scale sequence
        sequence_scaled = model_loader.sequence_scaler.transform(sequence.reshape(-1, 1)).reshape(sequence.shape)
        
        # Convert to tensor
        sequence_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).to(model_loader.device)
        
        # GRU Predict
        with torch.no_grad():
            prediction_scaled = model_loader.gru_model(sequence_tensor)
            gru_prediction = model_loader.target_scaler.inverse_transform(prediction_scaled.cpu().numpy().reshape(-1, 1))
        
        gru_prediction = float(gru_prediction[0, 0])
        print(f"✅ GRU prediction: {gru_prediction:,.2f}")
        
        # ========== STEP 2: XGBOOST ADJUSTMENT ==========
        print("🌳 Step 2: XGBoost adjustment...")
        
        # Prepare external features
        if external_factors_current is None:
            # Default external factors hiện tại
            external_factors_current = {
                'Temperature': 20.0,
                'Fuel_Price': 3.50,
                'CPI': 200.0,
                'Unemployment': 5.0,
                'Holiday_Flag': 0,
                'Month': datetime.now().month,
                'WeekOfYear': datetime.now().isocalendar()[1],
                'Year': datetime.now().year,
                'DayOfWeek': datetime.now().weekday(),
                'Is_Weekend': 1 if datetime.now().weekday() >= 5 else 0
            }
        
        if external_factors_previous is None:
            # Default external factors tuần trước (để tính change)
            external_factors_previous = {
                'Temperature': 20.0,
                'Fuel_Price': 3.50,
                'CPI': 200.0,
                'Unemployment': 5.0
            }
        
        # Prepare features for XGBoost với tính toán change
        features_array = prepare_external_features_with_changes(external_factors_current, external_factors_previous, model_loader.feature_columns)
        
        # XGBoost predict adjustment
        xgb_adjustment = model_loader.xgb_model.predict(features_array)[0]
        print(f"✅ XGBoost adjustment: {xgb_adjustment:,.2f}")
        
        # ========== STEP 3: ENSEMBLE FINAL PREDICTION ==========
        print("🚀 Step 3: Ensemble final prediction...")
        
        final_prediction = gru_prediction + xgb_adjustment
        print(f"✅ Final prediction: {final_prediction:,.2f}")
        
        # ========== STEP 4: CALCULATE CONFIDENCE ==========
        # Base confidence từ GRU
        input_std = np.std(sales_history)
        input_mean = np.mean(sales_history)
        cv = input_std / input_mean if input_mean > 0 else 0
        base_confidence = max(0.5, 1 - cv)
        
        # Adjust confidence based on adjustment magnitude
        adjustment_ratio = abs(xgb_adjustment) / gru_prediction if gru_prediction > 0 else 0
        if adjustment_ratio > 0.3:  # Large adjustment
            confidence = base_confidence * 0.7
        elif adjustment_ratio > 0.1:  # Medium adjustment
            confidence = base_confidence * 0.85
        else:  # Small adjustment
            confidence = base_confidence
        
        confidence = min(0.95, max(0.3, confidence))  # Clamp between 0.3 and 0.95
        
        # ========== STEP 5: PREPARE RESPONSE ==========
        message = f"Ensemble prediction successful - GRU: {gru_prediction:,.0f}, XGBoost adjustment: {xgb_adjustment:,.0f}"
        
        ensemble_breakdown = {
            "gru_contribution": float(gru_prediction),
            "xgboost_contribution": float(xgb_adjustment),
            "adjustment_ratio": float(adjustment_ratio),
            "prediction_method": "GRU_base + XGBoost_adjustment"
        }
        
        # ========== STEP 6: CREATE ADJUSTMENT EXPLANATION ==========
        adjustment_explanation = create_adjustment_explanation_with_shap(model_loader, features_array, external_factors_current, external_factors_previous, xgb_adjustment)
        
        return {
            "gru_prediction": gru_prediction,
            "xgboost_adjustment": xgb_adjustment,
            "final_prediction": final_prediction,
            "confidence_score": confidence,
            "input_sequence": sales_history,
            "external_factors_current": external_factors_current,
            "external_factors_previous": external_factors_previous,
            "message": message,
            "ensemble_breakdown": ensemble_breakdown,
            "adjustment_explanation": adjustment_explanation
        }
        
    except Exception as e:
        print(f"❌ Error in ensemble prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Lỗi dự đoán ensemble: {str(e)}")

# ========== 6. FASTAPI APP ==========
app = FastAPI(
    title="Ensemble GRU + XGBoost Sales Prediction API",
    description="API kết hợp GRU prediction + XGBoost adjustment cho doanh thu Walmart",
    version="1.0.0"
)

# Load models khi khởi động
print("🚀 Initializing Ensemble API...")
model_loader = EnsembleModelLoader()

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Ensemble GRU + XGBoost Sales Prediction API",
        "version": "1.0.0",
        "models": {
            "gru": "Improved GRU (Bidirectional + Attention)",
            "xgboost": "XGBoost Adjustment Model"
        },
        "architecture": "GRU_base + XGBoost_adjustment = Final_Prediction",
        "input_features": {
            "gru": "10 tuần doanh thu trước",
            "xgboost": "External factors (temperature, fuel price, etc.)"
        },
        "output": "Doanh thu tuần tiếp theo với ensemble prediction"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gru_loaded": model_loader.gru_model is not None,
        "xgboost_loaded": model_loader.xgb_model is not None,
        "device": str(model_loader.device),
        "feature_columns": len(model_loader.feature_columns) if model_loader.feature_columns else 0
    }

@app.post("/predict", response_model=EnsemblePredictionResponse)
def predict_sales_ensemble_endpoint(request: EnsemblePredictionRequest):
    """
    Dự đoán doanh thu tuần tiếp theo với ensemble GRU + XGBoost
    
    **Input:**
    - sales_history: List 10 giá trị doanh thu tuần trước
    - external_factors: Optional dict với external factors
    
    **Output:**
    - gru_prediction: Dự đoán từ GRU model
    - xgboost_adjustment: Điều chỉnh từ XGBoost
    - final_prediction: Kết quả cuối cùng
    - confidence_score: Độ tin cậy
    """
    try:
        result = predict_sales_ensemble(model_loader, request.sales_history, request.external_factors_current, request.external_factors_previous)
        return EnsemblePredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def get_model_info():
    """Thông tin về ensemble model"""
    return {
        "ensemble_type": "GRU + XGBoost",
        "architecture": {
            "gru": {
                "type": "Improved GRU",
                "input_size": 1,
                "hidden_size": 128,
                "num_layers": 2,
                "bidirectional": True,
                "attention": True
            },
            "xgboost": {
                "type": "XGBoost Adjustment",
                "features": len(model_loader.feature_columns) if model_loader.feature_columns else 0,
                "feature_type": "Relative Changes"
            }
        },
        "prediction_flow": {
            "step1": "GRU predicts base sales from 10-week history",
            "step2": "XGBoost predicts adjustment from external factors",
            "step3": "Final = GRU_pred + XGBoost_adjustment"
        },
        "lookback_period": 10,
        "external_factors": model_loader.feature_columns if model_loader.feature_columns else [],
        "description": "Ensemble model kết hợp GRU base prediction và XGBoost adjustment"
    }

@app.get("/example")
def get_example():
    """Ví dụ input cho Ensemble API"""
    return {
        "example_request": {
            "sales_history": [1000000, 1050000, 1100000, 1150000, 1200000, 1250000, 1300000, 1350000, 1400000, 1450000],
            "external_factors_current": {
                "Temperature": 28.0,
                "Fuel_Price": 3.60,
                "CPI": 203.0,
                "Unemployment": 4.95,
                "Holiday_Flag": 1,
                "Month": 12,
                "WeekOfYear": 50,
                "Year": 2024,
                "DayOfWeek": 2,
                "Is_Weekend": 0
            },
            "external_factors_previous": {
                "Temperature": 25.0,
                "Fuel_Price": 3.50,
                "CPI": 200.0,
                "Unemployment": 5.0
            }
        },
        "expected_output": {
            "gru_prediction": "Base prediction from GRU",
            "xgboost_adjustment": "Adjustment from external factors",
            "final_prediction": "GRU_pred + XGBoost_adjustment"
        },
        "note": "API sẽ tự động tính Temperature_change, Fuel_Price_change, etc. từ external_factors_previous và external_factors_current"
    }

@app.get("/test-ensemble")
def test_ensemble_workflow():
    """Test luồng hoạt động ensemble với các scenarios khác nhau"""
    
    test_scenarios = {
        "increasing_trend": {
            "sales_history": [1000000, 1050000, 1100000, 1150000, 1200000, 1250000, 1300000, 1350000, 1400000, 1450000],
            "external_factors_current": {
                "Temperature": 28.0,
                "Fuel_Price": 3.60,
                "CPI": 203.0,
                "Unemployment": 4.95,
                "Holiday_Flag": 1,
                "Month": 12,
                "WeekOfYear": 50,
                "Year": 2024,
                "DayOfWeek": 2,
                "Is_Weekend": 0
            },
            "external_factors_previous": {
                "Temperature": 25.0,
                "Fuel_Price": 3.50,
                "CPI": 200.0,
                "Unemployment": 5.0
            }
        },
        "decreasing_trend": {
            "sales_history": [1500000, 1450000, 1400000, 1350000, 1300000, 1250000, 1200000, 1150000, 1100000, 1050000],
            "external_factors_current": {
                "Temperature": 17.0,
                "Fuel_Price": 3.78,
                "CPI": 198.0,
                "Unemployment": 5.1,
                "Holiday_Flag": 0,
                "Month": 1,
                "WeekOfYear": 3,
                "Year": 2024,
                "DayOfWeek": 1,
                "Is_Weekend": 0
            },
            "external_factors_previous": {
                "Temperature": 20.0,
                "Fuel_Price": 3.50,
                "CPI": 200.0,
                "Unemployment": 5.0
            }
        },
        "stable_trend": {
            "sales_history": [1200000, 1200000, 1200000, 1200000, 1200000, 1200000, 1200000, 1200000, 1200000, 1200000],
            "external_factors_current": {
                "Temperature": 20.0,
                "Fuel_Price": 3.50,
                "CPI": 200.0,
                "Unemployment": 5.0,
                "Holiday_Flag": 0,
                "Month": 6,
                "WeekOfYear": 25,
                "Year": 2024,
                "DayOfWeek": 3,
                "Is_Weekend": 0
            },
            "external_factors_previous": {
                "Temperature": 20.0,
                "Fuel_Price": 3.50,
                "CPI": 200.0,
                "Unemployment": 5.0
            }
        }
    }
    
    results = {}
    for scenario_name, test_data in test_scenarios.items():
        try:
            print(f"\n🧪 Testing scenario: {scenario_name}")
            result = predict_sales_ensemble(model_loader, test_data["sales_history"], test_data["external_factors_current"], test_data["external_factors_previous"])
            
            results[scenario_name] = {
                "input_trend": test_data["sales_history"][-3:],  # Last 3 values
                "gru_prediction": result["gru_prediction"],
                "xgboost_adjustment": result["xgboost_adjustment"],
                "final_prediction": result["final_prediction"],
                "confidence": result["confidence_score"],
                "adjustment_ratio": result["ensemble_breakdown"]["adjustment_ratio"]
            }
            
            print(f"✅ {scenario_name}: GRU={result['gru_prediction']:,.0f}, XGB={result['xgboost_adjustment']:,.0f}, Final={result['final_prediction']:,.0f}")
            
        except Exception as e:
            results[scenario_name] = {"error": str(e)}
            print(f"❌ {scenario_name}: {str(e)}")
    
    return {
        "ensemble_tests": results,
        "note": "Test luồng hoạt động ensemble với các scenarios khác nhau",
        "workflow": "GRU → XGBoost → Ensemble Final"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Ensemble GRU + XGBoost Sales Prediction API...")
    print("📊 Models: GRU (Base) + XGBoost (Adjustment)")
    print("🎯 Endpoint: http://localhost:8000")
    print("📖 Documentation: http://localhost:8000/docs")
    print("🧪 Test endpoint: http://localhost:8000/test-ensemble")
    uvicorn.run(app, host="0.0.0.0", port=8000)
