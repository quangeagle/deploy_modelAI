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

class ComparisonResponse(BaseModel):
    """Output model cho Comparison API"""
    # GRU Standalone results
    gru_standalone: dict
    # GRU + XGBoost Ensemble results
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
def predict_gru_standalone(model_loader, sales_history: List[float]) -> dict:
    """Dự đoán chỉ với GRU model"""
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
        
        # Calculate confidence score
        input_std = np.std(sales_history)
        input_mean = np.mean(sales_history)
        cv = input_std / input_mean if input_mean > 0 else 0
        confidence = max(0.5, 1 - cv)
        
        return {
            "predicted_sales": raw_prediction,
            "confidence_score": confidence,
            "input_sequence": sales_history,
            "message": "GRU standalone prediction successful",
            "method": "GRU_only",
            "model_type": "Improved GRU (Bidirectional + Attention)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi GRU prediction: {str(e)}")

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
    """Dự đoán với GRU + XGBoost ensemble"""
    try:
        print("🚀 Starting ensemble prediction...")
        
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
        
        # ========== STEP 2: XGBOOST ADJUSTMENT ==========
        print("🔍 Step 2: XGBoost adjustment...")
        
        # Prepare external features
        features_array = prepare_external_features_with_changes(
            external_factors_current, external_factors_previous, model_loader.feature_columns
        )
        
        # Predict adjustment with XGBoost
        xgb_adjustment = float(model_loader.xgb_model.predict(features_array)[0])
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
            "method": "GRU_Ensemble",
            "model_type": "GRU + XGBoost Ensemble"
        }
        
    except Exception as e:
        print(f"❌ Error in ensemble prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Lỗi dự đoán ensemble: {str(e)}")

# ========== 6. COMPARISON ANALYSIS ==========
def analyze_comparison(gru_standalone: dict, gru_ensemble: dict) -> dict:
    """Phân tích so sánh giữa 2 phương pháp"""
    try:
        # Extract predictions
        gru_pred = gru_standalone["predicted_sales"]
        ensemble_pred = gru_ensemble["final_prediction"]
        xgb_adjustment = gru_ensemble["xgboost_adjustment"]
        
        # Calculate differences
        absolute_difference = abs(ensemble_pred - gru_pred)
        relative_difference = (absolute_difference / gru_pred) * 100 if gru_pred > 0 else 0
        
        # Determine which method gives higher prediction
        higher_method = "Ensemble" if ensemble_pred > gru_pred else "GRU_Standalone"
        lower_method = "GRU_Standalone" if ensemble_pred > gru_pred else "Ensemble"
        
        # Calculate confidence difference
        confidence_diff = gru_ensemble["confidence_score"] - gru_standalone["confidence_score"]
        
        # Determine recommendation
        if abs(relative_difference) < 5:  # Less than 5% difference
            recommendation = "Both methods give similar results. GRU standalone is sufficient for basic predictions."
        elif abs(relative_difference) < 15:  # 5-15% difference
            recommendation = "Moderate difference. Consider using ensemble for more accurate predictions with external factors."
        else:  # More than 15% difference
            recommendation = "Significant difference. Ensemble method provides substantial improvement by considering external factors."
        
        # Impact analysis
        if xgb_adjustment > 0:
            impact_description = f"XGBoost adjustment increases prediction by {xgb_adjustment:,.0f} ({relative_difference:.1f}%)"
        elif xgb_adjustment < 0:
            impact_description = f"XGBoost adjustment decreases prediction by {abs(xgb_adjustment):,.0f} ({relative_difference:.1f}%)"
        else:
            impact_description = "XGBoost adjustment has no effect"
        
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
                "gru_standalone_confidence": gru_standalone["confidence_score"],
                "ensemble_confidence": gru_ensemble["confidence_score"],
                "confidence_difference": confidence_diff
            },
            "xgboost_impact": {
                "adjustment_value": xgb_adjustment,
                "adjustment_percentage": relative_difference,
                "impact_description": impact_description
            },
            "recommendation": recommendation,
            "method_analysis": {
                "gru_standalone": "Uses only historical sales data for prediction",
                "ensemble": "Combines historical sales (GRU) with external factors (XGBoost)",
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
        "message": "Comparison API: GRU Standalone vs GRU+XGBoost Ensemble",
        "version": "1.0.0",
        "models": {
            "gru": "Improved GRU (Bidirectional + Attention)",
            "xgboost": "XGBoost Adjustment Model"
        },
        "comparison_methods": {
            "method1": "GRU Standalone (historical sales only)",
            "method2": "GRU + XGBoost Ensemble (historical + external factors)"
        },
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
                "xgboost_adjustment": 0,
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
        
        return ComparisonResponse(
            gru_standalone=gru_standalone_result,
            gru_ensemble=gru_ensemble_result,
            comparison_analysis=comparison_result,
            input_data=input_data
        )
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        raise HTTPException(status_code=400, detail=f"Lỗi so sánh: {str(e)}")

@app.post("/gru-standalone")
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
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/example")
def get_example():
    """Ví dụ input cho comparison API"""
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
        "note": "Gửi POST request đến /compare để so sánh cả 2 phương pháp"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Comparison API...")
    print("📊 Models: GRU Standalone vs GRU+XGBoost Ensemble")
    print("🎯 Endpoint: http://localhost:8000")
    print("📖 Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
