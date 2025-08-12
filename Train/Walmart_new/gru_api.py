# GRU Sales Prediction API
# FastAPI endpoint để test mô hình GRU

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

# ========== 1. GRU MODEL CLASS ==========
class GRUSalesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(GRUSalesPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.linear = nn.Linear(hidden_size, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        gru_out, _ = self.gru(x)
        
        # Lấy output của timestep cuối cùng
        last_output = gru_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Linear layer để dự đoán
        output = self.linear(last_output)
        
        return output

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

# ========== 3. MODEL LOADER ==========
class GRUModelLoader:
    def __init__(self):
        self.model = None
        self.sequence_scaler = None
        self.target_scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load trained GRU model và scalers"""
        try:
            print("🔄 Loading GRU model...")
            
            # Load model
            self.model = GRUSalesPredictor(input_size=1, hidden_size=128, num_layers=2, dropout=0.2)
            self.model.load_state_dict(torch.load('best_gru_model.pth', map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            # Load scalers
            with open('sequence_scaler.pkl', 'rb') as f:
                self.sequence_scaler = pickle.load(f)
            
            with open('target_scaler.pkl', 'rb') as f:
                self.target_scaler = pickle.load(f)
            
            print("✅ GRU model loaded successfully!")
            print(f"🔧 Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

# ========== 4. PREDICTION FUNCTION ==========
def predict_sales(model_loader, sales_history: List[float]) -> dict:
    """
    Dự đoán doanh thu dựa trên lịch sử doanh thu
    """
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
            prediction_scaled = model_loader.model(sequence_tensor)
            prediction = model_loader.target_scaler.inverse_transform(prediction_scaled.cpu().numpy().reshape(-1, 1))
        
        predicted_sales = float(prediction[0, 0])
        
        # Calculate confidence score (dựa trên độ ổn định của input)
        input_std = np.std(sales_history)
        input_mean = np.mean(sales_history)
        cv = input_std / input_mean if input_mean > 0 else 0
        confidence = max(0.5, 1 - cv)  # Confidence cao hơn khi input ổn định
        
        return {
            "predicted_sales": predicted_sales,
            "confidence_score": confidence,
            "input_sequence": sales_history,
            "message": "Dự đoán thành công"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi dự đoán: {str(e)}")

# ========== 5. FASTAPI APP ==========
app = FastAPI(
    title="GRU Sales Prediction API",
    description="API để dự đoán doanh thu tuần tiếp theo dựa trên doanh thu 10 tuần trước",
    version="1.0.0"
)

# Load model khi khởi động
model_loader = GRUModelLoader()

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "GRU Sales Prediction API",
        "version": "1.0.0",
        "model": "GRU (Weekly Sales Only)",
        "input_features": "10 tuần doanh thu trước",
        "output": "Doanh thu tuần tiếp theo"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loader.model is not None,
        "device": str(model_loader.device)
    }

@app.post("/predict", response_model=SalesPredictionResponse)
def predict_sales_endpoint(request: SalesPredictionRequest):
    """
    Dự đoán doanh thu tuần tiếp theo
    
    **Input:**
    - sales_history: List 10 giá trị doanh thu tuần trước
    
    **Output:**
    - predicted_sales: Doanh thu dự đoán
    - confidence_score: Độ tin cậy (0-1)
    - input_sequence: Chuỗi input
    - message: Thông báo
    """
    try:
        result = predict_sales(model_loader, request.sales_history)
        return SalesPredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def get_model_info():
    """Thông tin về model"""
    return {
        "model_type": "GRU",
        "input_size": 1,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "lookback_period": 10,
        "features": ["Weekly_Sales"],
        "training_metrics": {
            "r2_score": 0.9650,
            "rmse": 57409.36,
            "mae": 36646.54
        },
        "description": "GRU model chỉ sử dụng doanh thu 10 tuần trước để dự đoán doanh thu tuần tiếp theo"
    }

@app.get("/example")
def get_example():
    """Ví dụ input cho API"""
    return {
        "example_request": {
            "sales_history": [
                1000000,  # Tuần 1
                1050000,  # Tuần 2
                1100000,  # Tuần 3
                1150000,  # Tuần 4
                1200000,  # Tuần 5
                1250000,  # Tuần 6
                1300000,  # Tuần 7
                1350000,  # Tuần 8
                1400000,  # Tuần 9
                1450000   # Tuần 10
            ]
        },
        "expected_response": {
            "predicted_sales": 1500000,
            "confidence_score": 0.85,
            "input_sequence": [1000000, 1050000, 1100000, 1150000, 1200000, 1250000, 1300000, 1350000, 1400000, 1450000],
            "message": "Dự đoán thành công"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting GRU Sales Prediction API...")
    print("📊 Model: GRU (Weekly Sales Only)")
    print("🎯 Endpoint: http://localhost:8000")
    print("📖 Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
