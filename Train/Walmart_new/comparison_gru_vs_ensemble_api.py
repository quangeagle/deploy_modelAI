# Comparison API: GRU Standalone vs GRU+XGBoost Ensemble
# So sánh kết quả dự đoán giữa 2 phương pháp

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
class ComparisonRequest(BaseModel):
    """Input model cho Comparison API"""
    sales_history: List[float]  # List 10 giá trị doanh thu tuần trước
    external_factors_current: Optional[dict] = None  # External factors tuần hiện tại
    external_factors_previous: Optional[dict] = None  # External factors tuần trước

class SalesPredictionResponse(BaseModel):
    """Output model cho GRU API (giống improved_gru_api.py)"""
    predicted_sales: float
    confidence_score: float
    input_sequence: List[float]
    message: str
    trend_detected: str
    was_adjusted: bool
    raw_prediction: float
    adjusted_prediction: float

class XGBoostExplanationResponse(BaseModel):
    """Output model cho XGBoost explanation"""
    adjustment_value: float
    adjustment_percentage: float
    feature_contributions: dict
    top_positive_factors: List[dict]
    top_negative_factors: List[dict]
    business_insights: List[str]
    shap_analysis: dict

class ComparisonResponse(BaseModel):
    """Output model cho Comparison API"""
    # GRU Standalone results (sử dụng SalesPredictionResponse)
    gru_standalone: SalesPredictionResponse
    # GRU + XGBoost Ensemble results (đã bao gồm xgboost_explanation)
    gru_ensemble: dict
    # Comparison analysis
    comparison_analysis: dict
    # Input data
    input_data: dict

# ========== 3. MODEL LOADER ==========
class ComparisonModelLoader:
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
            print("🔄 Loading models for comparison...")
            
            # Load GRU model
            self.gru_model = ImprovedGRUSalesPredictor(input_size=1, hidden_size=128, num_layers=2, dropout=0.2)
            self.gru_model.load_state_dict(torch.load('improved_gru_model.pth', map_location=self.device))
            self.gru_model.to(self.device)
            self.gru_model.eval()
            
            # Load XGBoost model
            self.xgb_model = joblib.load('output_relative/xgb_model.pkl')
            
            # Load scalers
            with open('improved_sequence_scaler.pkl', 'rb') as f:
                self.sequence_scaler = pickle.load(f)
            
            with open('improved_target_scaler.pkl', 'rb') as f:
                self.target_scaler = pickle.load(f)
            
            # Load feature columns
            with open('output_relative/feature_columns.txt', 'r') as f:
                self.feature_columns = [line.strip() for line in f.readlines()]
            
            print("✅ All models loaded successfully!")
            print(f"🔧 Device: {self.device}")
            print(f"📊 Feature columns: {len(self.feature_columns)}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise e

# ========== 4. GRU STANDALONE PREDICTION ==========
def validate_trend_prediction(input_sequence, prediction):
    """
    Validate prediction dựa trên xu hướng input (giống improved_gru_api.py)
    """
    seq = np.asarray(input_sequence, dtype=float)
    n = len(seq)
    last = seq[-1]
    first = seq[0]

    # 1) Xu hướng dài hạn (so sánh đầu-cuối)
    overall_change = (last - first) / max(first, 1e-6)

    # 2) Trung bình động gần (5 tuần cuối vs 5 tuần trước)
    half = n // 2
    window = 5 if n >= 10 else max(2, n // 2)
    recent_mean = np.mean(seq[-window:])
    prev_mean = np.mean(seq[-2*window:-window]) if n >= 2*window else np.mean(seq[:half])
    ma_change = (recent_mean - prev_mean) / max(prev_mean, 1e-6)

    # 3) Tỉ lệ số bước giảm/tăng
    diffs = np.diff(seq)
    neg_ratio = float(np.mean(diffs < 0)) if diffs.size > 0 else 0.0
    pos_ratio = float(np.mean(diffs > 0)) if diffs.size > 0 else 0.0

    # 4) Slope chuẩn hóa (hồi quy tuyến tính)
    x = np.arange(n)
    slope = np.polyfit(x, seq, 1)[0] if n >= 2 else 0.0
    slope_norm = slope / max(np.mean(seq), 1e-6)

    # Phân loại xu hướng kết hợp nhiều tiêu chí
    trend_type = "volatile"
    if (overall_change <= -0.08) or (ma_change <= -0.05) or (slope_norm <= -0.02) or (neg_ratio >= 0.7):
        trend_type = "strong_decreasing"
    elif (overall_change <= -0.03) or (ma_change <= -0.02) or (slope_norm <= -0.01) or (neg_ratio >= 0.6):
        trend_type = "decreasing"
    elif (overall_change >= 0.08) or (ma_change >= 0.05) or (slope_norm >= 0.02) or (pos_ratio >= 0.7):
        trend_type = "strong_increasing"
    elif (overall_change >= 0.03) or (ma_change >= 0.02) or (slope_norm >= 0.01) or (pos_ratio >= 0.6):
        trend_type = "increasing"
    else:
        # Ổn định nếu biên độ nhỏ và bước lên/xuống cân bằng
        amplitude = (np.max(seq) - np.min(seq)) / max(np.mean(seq), 1e-6)
        if amplitude < 0.05 and 0.4 <= pos_ratio <= 0.6:
            trend_type = "stable"
        else:
            trend_type = "volatile"

    # Kiểm tra hướng dự đoán so với xu hướng phát hiện
    predicted_direction = 1 if prediction > last else -1 if prediction < last else 0
    expected_direction = 0
    if trend_type in ("strong_increasing", "increasing"):
        expected_direction = 1
    elif trend_type in ("strong_decreasing", "decreasing"):
        expected_direction = -1

    # Nếu xung đột hướng, điều chỉnh về cùng chiều với xu hướng
    if expected_direction != 0 and predicted_direction != 0 and expected_direction != predicted_direction:
        magnitude = max(abs(overall_change), abs(ma_change), abs(slope_norm), 0.02)
        adjust_factor = 0.6 * magnitude  # mức điều chỉnh 60% cường độ xu hướng
        if expected_direction > 0:
            adjusted_prediction = float(last * (1 + adjust_factor))
        else:
            adjusted_prediction = float(last * (1 - adjust_factor))
        return adjusted_prediction, True, trend_type

    return float(prediction), False, trend_type

def predict_gru_standalone(model_loader, sales_history: List[float]) -> SalesPredictionResponse:
    """Dự đoán chỉ với GRU model và trả về SalesPredictionResponse"""
    try:
        # Validate input
        if len(sales_history) != 10:
            raise ValueError("Cần đúng 10 giá trị doanh thu tuần trước")
        
        # Convert to numpy array
        sequence = np.array(sales_history, dtype=np.float32)
        
        # Reshape to (1, 10, 1) for batch processing
        sequence = sequence.reshape(1, -1, 1)
        
        # Scale sequence
        sequence_scaled = model_loader.sequence_scaler.transform(sequence.reshape(-1, 1)).reshape(sequence.shape)
        
        # Convert to tensor
        sequence_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).to(model_loader.device)
        
        # Predict
        with torch.no_grad():
            prediction_scaled = model_loader.gru_model(sequence_tensor)
            prediction = model_loader.target_scaler.inverse_transform(prediction_scaled.cpu().numpy().reshape(-1, 1))
        
        raw_prediction = float(prediction[0, 0])
        
        # Validate trend
        adjusted_prediction, was_adjusted, trend_type = validate_trend_prediction(sales_history, raw_prediction)
        
        # Calculate confidence score
        input_std = np.std(sales_history)
        input_mean = np.mean(sales_history)
        cv = input_std / input_mean if input_mean > 0 else 0
        base_confidence = max(0.5, 1 - cv)
        
        # Adjust confidence based on trend consistency
        if was_adjusted:
            confidence = base_confidence * 0.8  # Lower confidence if adjusted
        else:
            confidence = base_confidence
        
        # Determine message
        if was_adjusted:
            message = f"Dự đoán đã được điều chỉnh theo xu hướng {trend_type}"
        else:
            message = f"Dự đoán thành công - xu hướng {trend_type}"
        
        return SalesPredictionResponse(
            predicted_sales=adjusted_prediction,
            confidence_score=confidence,
            input_sequence=sales_history,
            message=message,
            trend_detected=trend_type,
            was_adjusted=was_adjusted,
            raw_prediction=raw_prediction,
            adjusted_prediction=adjusted_prediction
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi GRU prediction: {str(e)}")

def create_xgboost_explanation(model_loader, features_array: np.ndarray, external_factors_current: dict, external_factors_previous: dict, xgb_adjustment: float) -> XGBoostExplanationResponse:
    """
    Tạo giải thích chi tiết về XGBoost adjustment với SHAP analysis
    """
    try:
        print("🔍 Creating XGBoost explanation with SHAP...")
        
        # ========== STEP 1: TÍNH SHAP VALUES ==========
        explainer = shap.TreeExplainer(model_loader.xgb_model)
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
        
        # ========== STEP 2: PHÂN TÍCH FEATURE CONTRIBUTIONS ==========
        feature_contributions = {}
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
            
            feature_contributions[feature_name] = {
                "feature_value": float(feature_value),
                "shap_value": float(shap_value),
                "impact": impact,
                "direction": direction,
                "contribution_percentage": f"{contribution_pct:.1f}%",
                "absolute_contribution": float(abs(shap_value))
            }
        
        # ========== STEP 3: TOP POSITIVE/NEGATIVE FACTORS ==========
        top_positive = shap_df[shap_df['shap_value'] > 0].head(5)
        top_negative = shap_df[shap_df['shap_value'] < 0].head(5)
        
        top_positive_factors = []
        for _, row in top_positive.iterrows():
            top_positive_factors.append({
                "feature": row['feature'],
                "shap_value": float(row['shap_value']),
                "contribution": f"{(abs(row['shap_value']) / total_shap * 100):.1f}%"
            })
        
        top_negative_factors = []
        for _, row in top_negative.iterrows():
            top_negative_factors.append({
                "feature": row['feature'],
                "shap_value": float(row['shap_value']),
                "contribution": f"{(abs(row['shap_value']) / total_shap * 100):.1f}%"
            })
        
        # ========== STEP 4: BUSINESS INSIGHTS ==========
        business_insights = []
        
        # Temperature analysis
        if 'Temperature_change' in feature_contributions:
            temp_contrib = feature_contributions['Temperature_change']
            if temp_contrib['shap_value'] > 0:
                business_insights.append("Nhiệt độ thay đổi → Khách hàng thoải mái mua sắm → Tăng doanh thu")
            else:
                business_insights.append("Nhiệt độ thay đổi → Khách hàng ít ra ngoài → Giảm doanh thu")
        
        # Fuel Price analysis
        if 'Fuel_Price_change' in feature_contributions:
            fuel_contrib = feature_contributions['Fuel_Price_change']
            if fuel_contrib['shap_value'] > 0:
                business_insights.append("Giá xăng thay đổi → Chi phí vận chuyển thấp → Tăng doanh thu")
            else:
                business_insights.append("Giá xăng thay đổi → Chi phí vận chuyển cao → Giảm doanh thu")
        
        # CPI analysis
        if 'CPI_change' in feature_contributions:
            cpi_contrib = feature_contributions['CPI_change']
            if cpi_contrib['shap_value'] > 0:
                business_insights.append("CPI thay đổi → Sức mua tăng → Tăng doanh thu")
            else:
                business_insights.append("CPI thay đổi → Lạm phát cao → Sức mua giảm → Giảm doanh thu")
        
        # Unemployment analysis
        if 'Unemployment_change' in feature_contributions:
            unemp_contrib = feature_contributions['Unemployment_change']
            if unemp_contrib['shap_value'] > 0:
                business_insights.append("Thất nghiệp thay đổi → Sức mua tăng → Tăng doanh thu")
            else:
                business_insights.append("Thất nghiệp thay đổi → Sức mua giảm → Giảm doanh thu")
        
        # Holiday analysis
        if 'Holiday_Flag' in feature_contributions:
            holiday_contrib = feature_contributions['Holiday_Flag']
            if holiday_contrib['shap_value'] > 0:
                business_insights.append("Có ngày lễ → Khách hàng mua sắm nhiều → Tăng doanh thu")
            else:
                business_insights.append("Không có ngày lễ → Doanh thu bình thường")
        
        # Month/Season analysis
        if 'Month' in feature_contributions:
            month = int(feature_contributions['Month']['feature_value'])
            if month in [11, 12]:
                business_insights.append(f"Tháng {month} - Mùa mua sắm cuối năm → Doanh thu cao")
            elif month in [1, 2]:
                business_insights.append(f"Tháng {month} - Sau lễ, mùa thấp điểm → Doanh thu thấp")
        
        # Week of Year analysis
        if 'WeekOfYear' in feature_contributions:
            week = int(feature_contributions['WeekOfYear']['feature_value'])
            if week >= 50:
                business_insights.append(f"Tuần {week} - Cuối năm, mua sắm cao → Doanh thu tăng")
            elif week <= 5:
                business_insights.append(f"Tuần {week} - Đầu năm, sau lễ → Doanh thu thấp")
        
        # Year analysis
        if 'Year' in feature_contributions:
            year = int(feature_contributions['Year']['feature_value'])
            if year >= 2024:
                business_insights.append(f"Năm {year} - Kinh tế phục hồi → Doanh thu tăng")
            else:
                business_insights.append(f"Năm {year} - Kinh tế ổn định → Doanh thu ổn định")
        
        # ========== STEP 5: SHAP ANALYSIS SUMMARY ==========
        shap_analysis = {
            "total_positive_contribution": float(shap_df[shap_df['shap_value'] > 0]['shap_value'].sum()),
            "total_negative_contribution": float(shap_df[shap_df['shap_value'] < 0]['shap_value'].sum()),
            "net_contribution": float(shap_df['shap_value'].sum()),
            "feature_ranking": shap_df[['feature', 'shap_value', 'abs_shap']].to_dict('records'),
            "total_features_analyzed": len(shap_df)
        }
        
        # ========== STEP 6: CALCULATE ADJUSTMENT PERCENTAGE ==========
        # Lấy GRU prediction để tính percentage
        adjustment_percentage = 0.0
        if 'gru_prediction' in locals():
            adjustment_percentage = (xgb_adjustment / gru_prediction * 100) if gru_prediction > 0 else 0
        
        return XGBoostExplanationResponse(
            adjustment_value=float(xgb_adjustment),
            adjustment_percentage=float(adjustment_percentage),
            feature_contributions=feature_contributions,
            top_positive_factors=top_positive_factors,
            top_negative_factors=top_negative_factors,
            business_insights=business_insights,
            shap_analysis=shap_analysis
        )
        
    except Exception as e:
        print(f"❌ Error creating XGBoost explanation: {e}")
        # Return basic explanation if SHAP fails
        return XGBoostExplanationResponse(
            adjustment_value=float(xgb_adjustment),
            adjustment_percentage=0.0,
            feature_contributions={},
            top_positive_factors=[],
            top_negative_factors=[],
            business_insights=["Không thể tạo SHAP analysis"],
            shap_analysis={}
        )

# ========== 5. GRU + XGBOOST ENSEMBLE PREDICTION ==========
def prepare_external_features_with_changes(external_factors_current: dict, external_factors_previous: dict, feature_columns: List[str]) -> np.ndarray:
    """Chuẩn bị external features cho XGBoost với tính toán thay đổi tương đối"""
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
                    features_df[col] = 1  # Default values
                else:
                    features_df[col] = 0.0
        
        # Đảm bảo thứ tự columns đúng
        features_array = features_df[feature_columns].values.astype(np.float32)
        
        return features_array
        
    except Exception as e:
        print(f"❌ Error preparing features: {e}")
        raise e

def predict_gru_ensemble(model_loader, sales_history: List[float], external_factors_current: dict, external_factors_previous: dict) -> dict:
    """Dự đoán với GRU + XGBoost ensemble sử dụng adjustment ratio"""
    try:
        print("🚀 Starting ensemble prediction with adjustment ratio...")
        
        # ========== STEP 1: GRU PREDICTION ==========
        print("🔍 Step 1: GRU base prediction...")
        
        # Convert to numpy array
        sequence = np.array(sales_history, dtype=np.float32)
        sequence = sequence.reshape(1, -1, 1)
        
        # Scale sequence
        sequence_scaled = model_loader.sequence_scaler.transform(sequence.reshape(-1, 1)).reshape(sequence.shape)
        
        # Convert to tensor
        sequence_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).to(model_loader.device)
        
        # Predict with GRU
        with torch.no_grad():
            prediction_scaled = model_loader.gru_model(sequence_tensor)
            prediction = model_loader.target_scaler.inverse_transform(prediction_scaled.cpu().numpy().reshape(-1, 1))
        
        gru_prediction = float(prediction[0, 0])
        print(f"✅ GRU prediction: {gru_prediction:,.2f}")
        
        # ========== STEP 2: XGBOOST ADJUSTMENT RATIO ==========
        print("🔍 Step 2: XGBoost adjustment ratio...")
        
        # Prepare external features
        features_array = prepare_external_features_with_changes(
            external_factors_current, external_factors_previous, model_loader.feature_columns
        )
        
        # Predict adjustment ratio with XGBoost (đây là tỷ lệ, không phải giá trị tuyệt đối)
        xgb_adjustment_ratio = float(model_loader.xgb_model.predict(features_array)[0])
        print(f"✅ XGBoost adjustment ratio: {xgb_adjustment_ratio:.4f} ({xgb_adjustment_ratio*100:.2f}%)")
        
        # ========== STEP 3: ENSEMBLE FINAL PREDICTION (MỚI) ==========
        print("🚀 Step 3: Ensemble final prediction với adjustment ratio...")
        
        # 🚀 THAY ĐỔI CHÍNH: final_pred = GRU_pred * (1 + adjustment_ratio)
        # Thay vì: final_pred = GRU_pred + adjustment
        final_prediction = gru_prediction * (1 + xgb_adjustment_ratio)
        
        # Tính adjustment value tuyệt đối để hiển thị
        xgb_adjustment_value = gru_prediction * xgb_adjustment_ratio
        
        print(f"✅ Final prediction: {final_prediction:,.2f}")
        print(f"✅ Adjustment value: {xgb_adjustment_value:,.2f}")
        
        # ========== STEP 4: CALCULATE CONFIDENCE ==========
        # Base confidence từ GRU
        input_std = np.std(sales_history)
        input_mean = np.mean(sales_history)
        cv = input_std / input_mean if input_mean > 0 else 0
        base_confidence = max(0.5, 1 - cv)
        
        # Adjust confidence based on adjustment ratio magnitude
        adjustment_ratio_abs = abs(xgb_adjustment_ratio)
        if adjustment_ratio_abs > 0.3:  # Large adjustment (>30%)
            confidence = base_confidence * 0.7
        elif adjustment_ratio_abs > 0.1:  # Medium adjustment (10-30%)
            confidence = base_confidence * 0.85
        else:  # Small adjustment (<10%)
            confidence = base_confidence
        
        confidence = min(0.95, max(0.3, confidence))  # Clamp between 0.3 and 0.95
        
        # ========== STEP 5: PREPARE RESPONSE ==========
        message = f"Ensemble prediction successful - GRU: {gru_prediction:,.0f}, XGBoost adjustment: {xgb_adjustment_ratio*100:.2f}%"
        
        ensemble_breakdown = {
            "gru_contribution": float(gru_prediction),
            "xgboost_adjustment_ratio": float(xgb_adjustment_ratio),
            "xgboost_adjustment_value": float(xgb_adjustment_value),
            "adjustment_percentage": float(xgb_adjustment_ratio * 100),
            "prediction_method": "GRU_base * (1 + XGBoost_adjustment_ratio)"
        }
        
        # ========== STEP 6: CREATE XGBOOST EXPLANATION ==========
        # Truyền adjustment value để tạo explanation
        xgboost_explanation = create_xgboost_explanation(
            model_loader, features_array, external_factors_current, external_factors_previous, xgb_adjustment_value
        )
        
        return {
            "gru_prediction": gru_prediction,
            "xgboost_adjustment_ratio": xgb_adjustment_ratio,
            "xgboost_adjustment_value": xgb_adjustment_value,
            "final_prediction": final_prediction,
            "confidence_score": confidence,
            "input_sequence": sales_history,
            "external_factors_current": external_factors_current,
            "external_factors_previous": external_factors_previous,
            "message": message,
            "ensemble_breakdown": ensemble_breakdown,
            "method": "GRU_Ensemble_Adjustment_Ratio",
            "model_type": "GRU + XGBoost Adjustment Ratio Ensemble",
            "xgboost_explanation": xgboost_explanation
        }
        
    except Exception as e:
        print(f"❌ Error in ensemble prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Lỗi dự đoán ensemble: {str(e)}")

# ========== 6. COMPARISON ANALYSIS ==========
def analyze_comparison(gru_standalone: SalesPredictionResponse, gru_ensemble: dict) -> dict:
    """Phân tích so sánh giữa 2 phương pháp"""
    try:
        # Extract predictions
        gru_pred = gru_standalone.predicted_sales
        ensemble_pred = gru_ensemble["final_prediction"]
        xgb_adjustment_value = gru_ensemble["xgboost_adjustment_value"]
        xgb_adjustment_ratio = gru_ensemble["xgboost_adjustment_ratio"]
        
        # Calculate differences
        absolute_difference = abs(ensemble_pred - gru_pred)
        relative_difference = (absolute_difference / gru_pred) * 100 if gru_pred > 0 else 0
        
        # Determine which method gives higher prediction
        higher_method = "Ensemble" if ensemble_pred > gru_pred else "GRU_Standalone"
        lower_method = "GRU_Standalone" if ensemble_pred > gru_pred else "Ensemble"
        
        # Calculate confidence difference
        confidence_diff = gru_ensemble["confidence_score"] - gru_standalone.confidence_score
        
        # Determine recommendation
        if abs(relative_difference) < 5:  # Less than 5% difference
            recommendation = "Both methods give similar results. GRU standalone is sufficient for basic predictions."
        elif abs(relative_difference) < 15:  # 5-15% difference
            recommendation = "Moderate difference. Consider using ensemble for more accurate predictions with external factors."
        else:  # More than 15% difference
            recommendation = "Significant difference. Ensemble method provides substantial improvement by considering external factors."
        
        # Impact analysis với adjustment ratio
        if xgb_adjustment_ratio > 0:
            impact_description = f"XGBoost adjustment ratio: +{xgb_adjustment_ratio*100:.2f}% → Tăng doanh thu {xgb_adjustment_value:,.0f}"
        elif xgb_adjustment_ratio < 0:
            impact_description = f"XGBoost adjustment ratio: {xgb_adjustment_ratio*100:.2f}% → Giảm doanh thu {abs(xgb_adjustment_value):,.0f}"
        else:
            impact_description = "XGBoost adjustment ratio: 0% → Không thay đổi doanh thu"
        
        # XGBoost explanation summary (if available)
        xgboost_summary = {}
        if "xgboost_explanation" in gru_ensemble:
            xgb_exp = gru_ensemble["xgboost_explanation"]
            xgboost_summary = {
                "top_positive_factors": xgb_exp.top_positive_factors[:3],  # Top 3 positive
                "top_negative_factors": xgb_exp.top_negative_factors[:3],  # Top 3 negative
                "business_insights": xgb_exp.business_insights,  # Lấy toàn bộ business insights từ XGBoost explanation
                "total_features_analyzed": xgb_exp.shap_analysis.get("total_features_analyzed", 0)
            }
        
        return {
            "prediction_comparison": {
                "gru_standalone": gru_pred,
                "gru_ensemble": ensemble_pred,
                "absolute_difference": absolute_difference,
                "relative_difference_percent": relative_difference,
                "higher_method": higher_method,
                "lower_method": lower_method
            },
            "confidence_comparison": {
                "gru_standalone_confidence": gru_standalone.confidence_score,
                "ensemble_confidence": gru_ensemble["confidence_score"],
                "confidence_difference": confidence_diff
            },
            "xgboost_impact": {
                "adjustment_ratio": xgb_adjustment_ratio,
                "adjustment_percentage": xgb_adjustment_ratio * 100,
                "adjustment_value": xgb_adjustment_value,
                "impact_description": impact_description
            },
            "xgboost_explanation_summary": xgboost_summary,
            "recommendation": recommendation,
            "method_analysis": {
                "gru_standalone": "Uses only historical sales data for prediction",
                "ensemble": "Combines historical sales (GRU) with external factors (XGBoost adjustment ratio)",
                "advantage_ensemble": "More comprehensive by considering external factors like weather, economy, holidays",
                "advantage_standalone": "Faster prediction, simpler, good for basic forecasting"
            }
        }
        
    except Exception as e:
        print(f"❌ Error in comparison analysis: {e}")
        return {
            "error": f"Không thể phân tích so sánh: {str(e)}"
        }

# ========== 7. FASTAPI APP ==========
app = FastAPI(
    title="Comparison API: GRU Standalone vs GRU+XGBoost Ensemble",
    description="So sánh kết quả dự đoán giữa 2 phương pháp",
    version="1.0.0"
)

# Load models khi khởi động
print("🚀 Initializing Comparison API...")
model_loader = ComparisonModelLoader()

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Comparison API: GRU Standalone vs GRU+XGBoost Adjustment Ratio Ensemble",
        "version": "1.0.0",
        "models": {
            "gru": "Improved GRU (Bidirectional + Attention)",
            "xgboost": "XGBoost Adjustment Ratio Model (Relative Changes)"
        },
        "comparison_methods": {
            "method1": "GRU Standalone (historical sales only)",
            "method2": "GRU + XGBoost Adjustment Ratio Ensemble (historical * (1 + adjustment_ratio))"
        },
        "ensemble_formula": "Final Prediction = GRU_prediction * (1 + XGBoost_adjustment_ratio)",
        "adjustment_ratio_example": "adjustment_ratio = 0.05 → tăng 5%, adjustment_ratio = -0.03 → giảm 3%",
        "endpoints": {
            "/compare": "Compare both methods",
            "/gru-standalone": "GRU prediction only",
            "/gru-ensemble": "Ensemble prediction only"
        }
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

@app.post("/compare", response_model=ComparisonResponse)
def compare_predictions(request: ComparisonRequest):
    """
    So sánh dự đoán giữa GRU standalone và GRU+XGBoost ensemble
    
    **Input:**
    - sales_history: List 10 giá trị doanh thu tuần trước
    - external_factors_current: External factors tuần hiện tại (optional)
    - external_factors_previous: External factors tuần trước (optional)
    
    **Output:**
    - gru_standalone: Kết quả dự đoán chỉ với GRU
    - gru_ensemble: Kết quả dự đoán với ensemble
    - comparison_analysis: Phân tích so sánh chi tiết
    """
    try:
        print("🔄 Starting comparison...")
        
        # ========== STEP 1: GRU STANDALONE PREDICTION ==========
        print("🔍 Step 1: GRU standalone prediction...")
        gru_standalone_result = predict_gru_standalone(model_loader, request.sales_history)
        
        # ========== STEP 2: GRU ENSEMBLE PREDICTION ==========
        print("🔍 Step 2: GRU ensemble prediction...")
        if request.external_factors_current and request.external_factors_previous:
            gru_ensemble_result = predict_gru_ensemble(
                model_loader, 
                request.sales_history, 
                request.external_factors_current, 
                request.external_factors_previous
            )
        else:
            # Nếu không có external factors, chỉ dùng GRU
            gru_ensemble_result = {
                **gru_standalone_result,
                "method": "GRU_Only_No_External_Factors",
                "xgboost_adjustment_ratio": 0.0,
                "xgboost_adjustment_value": 0.0,
                "external_factors_current": {},
                "external_factors_previous": {}
            }
        
        # ========== STEP 3: COMPARISON ANALYSIS ==========
        print("🔍 Step 3: Comparison analysis...")
        comparison_result = analyze_comparison(gru_standalone_result, gru_ensemble_result)
        
        # ========== STEP 4: PREPARE FINAL RESPONSE ==========
        input_data = {
            "sales_history": request.sales_history,
            "external_factors_current": request.external_factors_current or {},
            "external_factors_previous": request.external_factors_previous or {}
        }
        
        # Prepare XGBoost explanation (if available)
        xgboost_explanation = None
        if "xgboost_explanation" in gru_ensemble_result:
            xgboost_explanation = gru_ensemble_result["xgboost_explanation"]
        
        return ComparisonResponse(
            gru_standalone=gru_standalone_result,
            gru_ensemble=gru_ensemble_result,
            comparison_analysis=comparison_result,
            input_data=input_data
        )
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        raise HTTPException(status_code=400, detail=f"Lỗi so sánh: {str(e)}")

@app.post("/gru-standalone", response_model=SalesPredictionResponse)
def gru_standalone_only(request: ComparisonRequest):
    """Chỉ dự đoán với GRU standalone"""
    try:
        result = predict_gru_standalone(model_loader, request.sales_history)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/gru-ensemble")
def gru_ensemble_only(request: ComparisonRequest):
    """Chỉ dự đoán với GRU ensemble"""
    try:
        if not request.external_factors_current or not request.external_factors_previous:
            raise HTTPException(status_code=400, detail="Cần external_factors_current và external_factors_previous cho ensemble prediction")
        
        result = predict_gru_ensemble(
            model_loader, 
            request.sales_history, 
            request.external_factors_current, 
            request.external_factors_previous
        )
        
        # Return result with XGBoost explanation
        return {
            "prediction": result,
            "xgboost_explanation": result.get("xgboost_explanation", None)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/example")
def get_example():
    """Ví dụ input cho comparison API với adjustment ratio"""
    return {
        "example_request": {
            "sales_history": [800000, 850000, 900000, 950000, 1000000, 1050000, 1100000, 1150000, 1200000, 1250000],
            "external_factors_current": {
                "Temperature": 30.0,
                "Fuel_Price": 3.80,
                "CPI": 205.0,
                "Unemployment": 4.8,
                "Holiday_Flag": 1,
                "Month": 12,
                "WeekOfYear": 51,
                "Year": 2024,
                "DayOfWeek": 1,
                "Is_Weekend": 0
            },
            "external_factors_previous": {
                "Temperature": 25.0,
                "Fuel_Price": 3.50,
                "CPI": 200.0,
                "Unemployment": 5.0
            }
        },
        "note": "Gửi POST request đến /compare để so sánh cả 2 phương pháp",
        "ensemble_formula": "Final Prediction = GRU_prediction * (1 + XGBoost_adjustment_ratio)",
        "adjustment_ratio_explanation": {
            "positive": "adjustment_ratio > 0 → Tăng doanh thu (ví dụ: 0.05 = tăng 5%)",
            "negative": "adjustment_ratio < 0 → Giảm doanh thu (ví dụ: -0.03 = giảm 3%)",
            "zero": "adjustment_ratio = 0 → Không thay đổi doanh thu"
        },
        "scaling_benefit": "Adjustment ratio sẽ scale theo magnitude của GRU prediction, phù hợp với mọi khoảng giá trị từ hàng nghìn đến hàng tỉ"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Comparison API with Adjustment Ratio...")
    print("📊 Models: GRU Standalone vs GRU+XGBoost Adjustment Ratio Ensemble")
    print("🎯 Ensemble Formula: Final = GRU_pred * (1 + adjustment_ratio)")
    print("💡 Adjustment Ratio: 0.05 = +5%, -0.03 = -3%")
    print("🌍 Endpoint: http://localhost:8000")
    print("📖 Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
