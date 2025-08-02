from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load mô hình Walmart
walmart_models = {
    'XGBoost': joblib.load('xgboost_model.pkl'),
    'LightGBM': joblib.load('lightgbm_model.pkl'),
    'CatBoost': joblib.load('catboost_model.pkl')
}
WALMART_FEATURES = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

# Load mô hình Favorita
favorita_model = joblib.load('lgbm_model.pkl')  # Ví dụ chỉ 1 mô hình
FAVORITA_FEATURES = [
    "store_nbr", "item_nbr", "family", "city", "state", "type",
    "transactions", "oil_price", "day", "month", "dayofweek", "is_weekend", "is_holiday"
]
  # Tùy bạn định nghĩa

@app.route('/')
def home():
    return 'API đa mô hình: Walmart & Favorita'

# ------------------- Walmart -------------------
@app.route('/predict_walmart', methods=['POST'])
def predict_walmart():
    try:
        data = request.get_json()
        model_name = data.get('model')
        if model_name not in walmart_models:
            return jsonify({'error': 'Model không hợp lệ. Chọn XGBoost, LightGBM hoặc CatBoost.'})

        input_features = []
        for feat in WALMART_FEATURES:
            value = data.get(feat)
            if value is None:
                return jsonify({'error': f'Missing value for "{feat}"'})
            input_features.append(float(value))

        model = walmart_models[model_name]
        prediction = model.predict([input_features])[0]

        return jsonify({'model': model_name, 'prediction': round(float(prediction), 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

# ------------------- Favorita -------------------
@app.route('/predict_favorita', methods=['POST'])
def predict_favorita():
    try:
        data = request.get_json()

        input_features = []
        for feat in FAVORITA_FEATURES:
            value = data.get(feat)
            if value is None:
                return jsonify({'error': f'Missing value for "{feat}"'})
            input_features.append(float(value))  # Nếu có categorical bạn cần xử lý encode trước

        prediction = favorita_model.predict([input_features])[0]

        return jsonify({'model': 'LightGBM_Favorita', 'prediction': round(float(prediction), 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
