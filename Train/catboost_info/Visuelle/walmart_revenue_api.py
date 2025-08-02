from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import json
from typing import List, Optional
import uvicorn

# ========== 1. Model Classes ==========
class WalmartDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.linear(out[:, -1, :])

# ========== 2. Data Models ==========
class PredictionRequest(BaseModel):
    weekly_sales: List[float]  # 10 tuần gần nhất
    holiday_flag: List[int]     # 10 tuần gần nhất
    temperature: List[float]     # 10 tuần gần nhất
    fuel_price: List[float]     # 10 tuần gần nhất
    cpi: List[float]           # 10 tuần gần nhất
    unemployment: List[float]   # 10 tuần gần nhất
    week_of_year: List[int]    # 10 tuần gần nhất
    month: List[int]           # 10 tuần gần nhất

class PredictionResponse(BaseModel):
    predicted_weekly_sales: float
    confidence_score: float
    model_info: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

# ========== 3. Global Variables ==========
app = FastAPI(
    title="Walmart Revenue Prediction API",
    description="API để dự đoán doanh thu tuần tiếp theo cho Walmart",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
input_scaler = None
target_scaler = None
device = None
model_loaded = False

# ========== 4. Model Loading Function ==========
def load_model():
    """Load model và scalers từ checkpoint"""
    global model, input_scaler, target_scaler, device, model_loaded
    
    try:
        checkpoint_dir = 'model_checkpoints'
        
        # Check if model files exist
        model_path = os.path.join(checkpoint_dir, "best_model.pth")
        input_scaler_path = os.path.join(checkpoint_dir, "input_scaler.pkl")
        target_scaler_path = os.path.join(checkpoint_dir, "target_scaler.pkl")
        
        if not all(os.path.exists(path) for path in [model_path, input_scaler_path, target_scaler_path]):
            raise FileNotFoundError("Model files not found")
        
        # Load scalers
        with open(input_scaler_path, 'rb') as f:
            input_scaler = pickle.load(f)
        with open(target_scaler_path, 'rb') as f:
            target_scaler = pickle.load(f)
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
        model = GRUModel(input_size=len(feature_cols)).to(device)
        
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        model_loaded = True
        print(f"✅ Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        model_loaded = False
        raise

# ========== 5. Prediction Function ==========
def create_sequences(data, lookback=12):
    """Tạo sequences cho GRU model"""
    sequences, targets = [], []
    for i in range(len(data) - lookback):
        seq = data[i:i+lookback]
        target = data[i+lookback, 0]  # Weekly_Sales
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def predict_revenue(request_data: PredictionRequest) -> PredictionResponse:
    """Dự đoán doanh thu tuần tiếp theo"""
    global model, input_scaler, target_scaler, device
    
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Validate input data
        if len(request_data.weekly_sales) != 10:
            raise ValueError("Must provide exactly 10 weeks of data")
        
        # Create DataFrame
        data = {
            'Weekly_Sales': request_data.weekly_sales,
            'Holiday_Flag': request_data.holiday_flag,
            'Temperature': request_data.temperature,
            'Fuel_Price': request_data.fuel_price,
            'CPI': request_data.cpi,
            'Unemployment': request_data.unemployment,
            'WeekOfYear': request_data.week_of_year,
            'Month': request_data.month
        }
        
        df = pd.DataFrame(data)
        
        # Scale features
        feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
        input_features = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
        weekly_sales_col = ['Weekly_Sales']
        
        # Scale input features
        scaled_inputs_only = input_scaler.transform(df[input_features])
        scaled_weekly_sales = target_scaler.transform(df[weekly_sales_col])
        
        # Combine scaled data
        scaled_input = np.concatenate([scaled_weekly_sales, scaled_inputs_only], axis=1)
        
        # Create tensor
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            y_pred_scaled = model(input_tensor).detach().cpu().numpy()
        
        # Inverse transform
        y_pred_real = target_scaler.inverse_transform(y_pred_scaled)[0, 0]
        
        # Calculate confidence (simple approach)
        confidence = 0.85  # Placeholder - could be improved with uncertainty estimation
        
        return PredictionResponse(
            predicted_weekly_sales=float(y_pred_real),
            confidence_score=confidence,
            model_info={
                "model_type": "GRU",
                "input_features": len(feature_cols),
                "device": str(device),
                "scaler_type": "MinMaxScaler"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# ========== 6. API Endpoints ==========
@app.on_event("startup")
async def startup_event():
    """Load model khi startup"""
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model at startup: {e}")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Walmart Revenue Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "Predict next week revenue",
            "/predict-file": "Predict from Excel file"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        device=str(device) if device else "unknown"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    """Dự đoán doanh thu tuần tiếp theo"""
    return predict_revenue(request)

@app.post("/predict-file", response_model=PredictionResponse)
async def predict_from_file(file: UploadFile = File(...)):
    """Dự đoán từ file Excel"""
    try:
        # Read Excel file
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="File must be Excel format")
        
        df = pd.read_excel(file.file)
        
        # Validate file structure
        expected_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
        if not all(col in df.columns for col in expected_cols):
            raise HTTPException(status_code=400, detail=f"File must contain columns: {expected_cols}")
        
        if len(df) != 10:
            raise HTTPException(status_code=400, detail="File must contain exactly 10 weeks of data")
        
        # Convert to request format
        request_data = PredictionRequest(
            weekly_sales=df['Weekly_Sales'].tolist(),
            holiday_flag=df['Holiday_Flag'].tolist(),
            temperature=df['Temperature'].tolist(),
            fuel_price=df['Fuel_Price'].tolist(),
            cpi=df['CPI'].tolist(),
            unemployment=df['Unemployment'].tolist(),
            week_of_year=df['WeekOfYear'].tolist(),
            month=df['Month'].tolist()
        )
        
        return predict_revenue(request_data)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")

@app.post("/reload-model")
async def reload_model():
    """Reload model từ checkpoint"""
    try:
        load_model()
        return {"message": "Model reloaded successfully", "model_loaded": model_loaded}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

# ========== 7. Run Server ==========
if __name__ == "__main__":
    uvicorn.run(
        "walmart_revenue_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )