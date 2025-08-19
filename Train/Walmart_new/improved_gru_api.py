# Improved GRU Sales Prediction API
# API với trend validation và enhanced prediction capabilities

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import warnings
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
        
        # Additional layers (phù hợp với avg_pool + max_pool => concat => hidden*4)
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
class SalesPredictionRequest(BaseModel):
    """Input model cho API"""
    sales_history: List[float]  # List 10 giá trị doanh thu tuần trước

class SalesPredictionResponse(BaseModel):
    """Output model cho API"""
    predicted_sales: float
    confidence_score: float
    input_sequence: List[float]
    message: str
    trend_detected: str
    was_adjusted: bool
    raw_prediction: float
    adjusted_prediction: float

# ========== 3. TREND VALIDATION FUNCTION ==========
def validate_trend_prediction(input_sequence, prediction):
    """
    Validate prediction dựa trên xu hướng input.
    Nhạy hơn với các chuỗi giảm/ tăng kiểu zig-zag:
    - Dùng xu hướng dài hạn (first vs last)
    - Dùng trung bình động (5 tuần cuối vs 5 tuần trước đó)
    - Dùng tỉ lệ số bước giảm/tăng
    - Dùng slope (hồi quy tuyến tính) đã chuẩn hóa
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

# ========== 4. MODEL LOADER ==========
class ImprovedGRUModelLoader:
    def __init__(self):
        self.model = None
        self.sequence_scaler = None
        self.target_scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load improved GRU model và scalers"""
        try:
            print("🔄 Loading Improved GRU model...")
            
            # Load model (khớp kiến trúc đã train: 128 hidden, 2 layers)
            self.model = ImprovedGRUSalesPredictor(input_size=1, hidden_size=128, num_layers=2, dropout=0.2)
            self.model.load_state_dict(torch.load('improved_gru_model.pth', map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            # Load scalers (RobustScaler)
            with open('improved_sequence_scaler.pkl', 'rb') as f:
                self.sequence_scaler = pickle.load(f)
            
            with open('improved_target_scaler.pkl', 'rb') as f:
                self.target_scaler = pickle.load(f)
            
            print("✅ Improved GRU model loaded successfully!")
            print(f"🔧 Device: {self.device}")
            print(f"📊 Sequence Scaler: {type(self.sequence_scaler).__name__}")
            print(f"📊 Target Scaler: {type(self.target_scaler).__name__}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

# ========== 5. PREDICTION FUNCTION ==========
def predict_sales_improved(model_loader, sales_history: List[float]) -> dict:
    """
    Dự đoán doanh thu với improved model và trend validation
    Hỗ trợ đa dạng giá trị từ hàng nghìn đến hàng tỉ
    """
    try:
        # Validate input
        if len(sales_history) != 10:
            raise ValueError("Cần đúng 10 giá trị doanh thu tuần trước")
        
        # Validate giá trị input
        sales_array = np.array(sales_history, dtype=np.float32)
        if np.any(sales_array <= 0):
            raise ValueError("Tất cả giá trị doanh thu phải dương (> 0)")
        
        # Kiểm tra range giá trị
        min_val = np.min(sales_array)
        max_val = np.max(sales_array)
        mean_val = np.mean(sales_array)
        
        print(f"📊 Input validation:")
        print(f"   • Min: {min_val:,.0f}")
        print(f"   • Max: {max_val:,.0f}")
        print(f"   • Mean: {mean_val:,.0f}")
        print(f"   • Range: {max_val/min_val:.1f}x")
        
        # Convert to numpy array
        sequence = sales_array.reshape(1, -1, 1)
        
        # Scale sequence với RobustScaler
        sequence_scaled = model_loader.sequence_scaler.transform(sequence.reshape(-1, 1)).reshape(sequence.shape)
        
        # Convert to tensor
        sequence_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).to(model_loader.device)
        
        # Predict
        with torch.no_grad():
            prediction_scaled = model_loader.model(sequence_tensor)
            prediction = model_loader.target_scaler.inverse_transform(prediction_scaled.cpu().numpy().reshape(-1, 1))
        
        raw_prediction = float(prediction[0, 0])
        
        # Validate trend
        adjusted_prediction, was_adjusted, trend_type = validate_trend_prediction(sales_history, raw_prediction)
        
        # Calculate confidence score (dựa trên độ ổn định của input và trend consistency)
        input_std = np.std(sales_history)
        input_mean = np.mean(sales_history)
        cv = input_std / input_mean if input_mean > 0 else 0
        
        # Base confidence
        base_confidence = max(0.5, 1 - cv)
        
        # Adjust confidence based on trend consistency
        if was_adjusted:
            confidence = base_confidence * 0.8  # Lower confidence if adjusted
        else:
            confidence = base_confidence
        
        # Additional confidence adjustment based on value range
        if min_val < 1000:  # Hàng nghìn
            confidence *= 0.95
        elif max_val > 1000000000:  # Hàng tỉ
            confidence *= 0.95
        elif max_val > 100000000:  # Hàng trăm triệu
            confidence *= 0.98
        
        confidence = min(0.95, max(0.3, confidence))  # Clamp between 0.3 and 0.95
        
        # Determine message
        if was_adjusted:
            message = f"Dự đoán đã được điều chỉnh theo xu hướng {trend_type}"
        else:
            message = f"Dự đoán thành công - xu hướng {trend_type}"
        
        # Add value range info to message
        if min_val < 1000:
            message += " (hàng nghìn)"
        elif max_val > 1000000000:
            message += " (hàng tỉ)"
        elif max_val > 100000000:
            message += " (hàng trăm triệu)"
        
        return {
            "predicted_sales": adjusted_prediction,
            "confidence_score": confidence,
            "input_sequence": sales_history,
            "message": message,
            "trend_detected": trend_type,
            "was_adjusted": was_adjusted,
            "raw_prediction": raw_prediction,
            "adjusted_prediction": adjusted_prediction,
            "input_statistics": {
                "min_value": float(min_val),
                "max_value": float(max_val),
                "mean_value": float(mean_val),
                "value_range_ratio": float(max_val/min_val),
                "coefficient_of_variation": float(cv)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi dự đoán: {str(e)}")

# ========== 6. FASTAPI APP ==========
app = FastAPI(
    title="Improved GRU Sales Prediction API",
    description="API với trend validation và enhanced prediction capabilities",
    version="2.0.0"
)

# Load model khi khởi động
model_loader = ImprovedGRUModelLoader()

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Improved GRU Sales Prediction API",
        "version": "2.0.0",
        "model": "Improved GRU (Bidirectional + Attention)",
        "features": ["Trend Validation", "Enhanced Architecture", "Balanced Training"],
        "input_features": "10 tuần doanh thu trước",
        "output": "Doanh thu tuần tiếp theo với trend validation"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loader.model is not None,
        "device": str(model_loader.device),
        "model_type": "Improved GRU (Bidirectional + Attention)"
    }

@app.post("/predict", response_model=SalesPredictionResponse)
def predict_sales_endpoint(request: SalesPredictionRequest):
    """
    Dự đoán doanh thu tuần tiếp theo với trend validation
    
    **Input:**
    - sales_history: List 10 giá trị doanh thu tuần trước
    
    **Output:**
    - predicted_sales: Doanh thu dự đoán (có thể được điều chỉnh)
    - confidence_score: Độ tin cậy (0-1)
    - input_sequence: Chuỗi input
    - message: Thông báo chi tiết
    - trend_detected: Loại xu hướng phát hiện
    - was_adjusted: Có được điều chỉnh không
    """
    try:
        result = predict_sales_improved(model_loader, request.sales_history)
        return SalesPredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def get_model_info():
    """Thông tin về improved model"""
    return {
        "model_type": "Improved GRU",
        "architecture": {
            "input_size": 1,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "bidirectional": True,
            "attention_heads": 8
        },
        "features": {
            "trend_validation": True,
            "balanced_training": True,
            "synthetic_data": True,
            "attention_mechanism": True,
            "diverse_value_ranges": True,
            "robust_scaling": True
        },
        "value_range_support": {
            "min_value": "Hàng nghìn (1,000+)",
            "max_value": "Hàng tỉ (1,000,000,000+)",
            "scaling_method": "RobustScaler",
            "outlier_handling": "Robust"
        },
        "lookback_period": 10,
        "features_used": ["Weekly_Sales"],
        "trend_types": [
            "strong_increasing",
            "increasing", 
            "stable",
            "decreasing",
            "strong_decreasing",
            "volatile"
        ],
        "description": "Improved GRU model với trend validation, enhanced architecture và khả năng xử lý đa dạng giá trị từ hàng nghìn đến hàng tỉ"
    }

@app.get("/example")
def get_example():
    """Ví dụ input cho API với các xu hướng và giá trị đa dạng"""
    return {
        "examples": {
            "increasing_trend_thousands": {
                "sales_history": [5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500],
                "expected_trend": "strong_increasing",
                "value_range": "Hàng nghìn"
            },
            "decreasing_trend_millions": {
                "sales_history": [1500000, 1450000, 1400000, 1350000, 1300000, 1250000, 1200000, 1150000, 1100000, 1050000],
                "expected_trend": "strong_decreasing",
                "value_range": "Hàng triệu"
            },
            "stable_trend_billions": {
                "sales_history": [500000000, 500000000, 500000000, 500000000, 500000000, 500000000, 500000000, 500000000, 500000000, 500000000],
                "expected_trend": "stable",
                "value_range": "Hàng trăm triệu"
            },
            "volatile_trend_tens_of_thousands": {
                "sales_history": [50000, 80000, 30000, 90000, 40000, 70000, 60000, 85000, 55000, 75000],
                "expected_trend": "volatile",
                "value_range": "Hàng chục nghìn"
            },
            "increasing_trend_hundreds_of_millions": {
                "sales_history": [100000000, 110000000, 120000000, 130000000, 140000000, 150000000, 160000000, 170000000, 180000000, 190000000],
                "expected_trend": "strong_increasing",
                "value_range": "Hàng trăm triệu"
            }
        },
        "note": "Model sẽ tự động detect trend và validate prediction cho tất cả các giá trị từ hàng nghìn đến hàng tỉ",
        "supported_ranges": [
            "Hàng nghìn (1,000 - 99,999)",
            "Hàng chục nghìn (10,000 - 999,999)",
            "Hàng trăm nghìn (100,000 - 9,999,999)",
            "Hàng triệu (1,000,000 - 99,999,999)",
            "Hàng chục triệu (10,000,000 - 999,999,999)",
            "Hàng trăm triệu (100,000,000 - 999,999,999)",
            "Hàng tỉ (1,000,000,000+)"
        ]
    }

@app.get("/test-trends")
def test_different_trends():
    """Test các xu hướng khác nhau với đa dạng giá trị"""
    trends = {
        "strong_increasing_thousands": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        "increasing_millions": [1000000, 1050000, 1100000, 1150000, 1200000, 1250000, 1300000, 1350000, 1400000, 1450000],
        "stable_billions": [500000000, 500000000, 500000000, 500000000, 500000000, 500000000, 500000000, 500000000, 500000000, 500000000],
        "decreasing_tens_of_thousands": [50000, 45000, 40000, 35000, 30000, 25000, 20000, 15000, 10000, 5000],
        "strong_decreasing_hundreds_of_millions": [200000000, 190000000, 180000000, 170000000, 160000000, 150000000, 140000000, 130000000, 120000000, 110000000],
        "volatile_mixed": [50000, 80000, 30000, 90000, 40000, 70000, 60000, 85000, 55000, 75000]
    }
    
    results = {}
    for trend_name, sales_history in trends.items():
        try:
            result = predict_sales_improved(model_loader, sales_history)
            results[trend_name] = {
                "input": sales_history[-3:],  # Last 3 values
                "input_range": f"{min(sales_history):,.0f} - {max(sales_history):,.0f}",
                "predicted": result["predicted_sales"],
                "trend_detected": result["trend_detected"],
                "was_adjusted": result["was_adjusted"],
                "confidence": result["confidence_score"],
                "input_statistics": result.get("input_statistics", {})
            }
        except Exception as e:
            results[trend_name] = {"error": str(e)}
    
    return {
        "trend_tests": results,
        "note": "Test các xu hướng khác nhau với đa dạng giá trị để validate model performance",
        "value_ranges_tested": [
            "Hàng nghìn (1,000+)",
            "Hàng chục nghìn (10,000+)", 
            "Hàng triệu (1,000,000+)",
            "Hàng trăm triệu (100,000,000+)"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Improved GRU Sales Prediction API...")
    print("📊 Model: Improved GRU (Bidirectional + Attention)")
    print("🎯 Endpoint: http://localhost:8000")
    print("📖 Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
