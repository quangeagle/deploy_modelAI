# Test Script cho GRU Sales Prediction API
# Script để test API với các scenarios khác nhau

import requests
import json
import time

# API Configuration
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health Check PASSED")
            print(f"   Status: {data['status']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            print(f"   Device: {data['device']}")
            return True
        else:
            print("❌ Health Check FAILED")
            return False
    except Exception as e:
        print(f"❌ Health Check ERROR: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n🔍 Testing Model Info...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        if response.status_code == 200:
            data = response.json()
            print("✅ Model Info PASSED")
            print(f"   Model Type: {data['model_type']}")
            print(f"   R² Score: {data['training_metrics']['r2_score']}")
            print(f"   RMSE: ${data['training_metrics']['rmse']:,.2f}")
            print(f"   MAE: ${data['training_metrics']['mae']:,.2f}")
            return True
        else:
            print("❌ Model Info FAILED")
            return False
    except Exception as e:
        print(f"❌ Model Info ERROR: {e}")
        return False

def test_example():
    """Test example endpoint"""
    print("\n🔍 Testing Example...")
    try:
        response = requests.get(f"{BASE_URL}/example")
        if response.status_code == 200:
            data = response.json()
            print("✅ Example PASSED")
            print(f"   Input Sequence: {len(data['example_request']['sales_history'])} values")
            print(f"   Expected Prediction: ${data['expected_response']['predicted_sales']:,.2f}")
            return True
        else:
            print("❌ Example FAILED")
            return False
    except Exception as e:
        print(f"❌ Example ERROR: {e}")
        return False

def test_prediction(sales_history, description=""):
    """Test prediction endpoint với sales history"""
    print(f"\n🔍 Testing Prediction {description}...")
    print(f"   Input: {sales_history}")
    
    try:
        payload = {
            "sales_history": sales_history
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prediction PASSED")
            print(f"   Predicted Sales: ${data['predicted_sales']:,.2f}")
            print(f"   Confidence Score: {data['confidence_score']:.3f}")
            print(f"   Message: {data['message']}")
            return data
        else:
            print(f"❌ Prediction FAILED: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Prediction ERROR: {e}")
        return None

def run_all_tests():
    """Chạy tất cả tests"""
    print("="*60)
    print("🧪 GRU SALES PREDICTION API TESTING")
    print("="*60)
    
    # Test 1: Health Check
    health_ok = test_health_check()
    
    # Test 2: Model Info
    info_ok = test_model_info()
    
    # Test 3: Example
    example_ok = test_example()
    
    # Test 4: Predictions với các scenarios khác nhau
    print("\n" + "="*60)
    print("🎯 PREDICTION TESTS")
    print("="*60)
    
    # Scenario 1: Tăng dần (trending up)
    trending_up = [1000000, 1050000, 1100000, 1150000, 1200000, 
                   1250000, 1300000, 1350000, 1400000, 1450000]
    test_prediction(trending_up, "(Trending Up)")
    
    # Scenario 2: Giảm dần (trending down)
    trending_down = [1500000, 1450000, 1400000, 1350000, 1300000,
                     1250000, 1200000, 1150000, 1100000, 1050000]
    test_prediction(trending_down, "(Trending Down)")
    
    # Scenario 3: Ổn định (stable)
    stable = [1200000, 1200000, 1200000, 1200000, 1200000,
              1200000, 1200000, 1200000, 1200000, 1200000]
    test_prediction(stable, "(Stable)")
    
    # Scenario 4: Biến động (volatile)
    volatile = [1000000, 1500000, 800000, 1600000, 900000,
                1400000, 1100000, 1300000, 1200000, 1400000]
    test_prediction(volatile, "(Volatile)")
    
    # Scenario 5: Seasonal pattern (mùa vụ)
    seasonal = [1000000, 1100000, 1200000, 1300000, 1400000,
                1500000, 1600000, 1700000, 1800000, 1900000]
    test_prediction(seasonal, "(Seasonal Pattern)")
    
    # Scenario 6: Holiday spike (đỉnh ngày lễ)
    holiday_spike = [1200000, 1250000, 1300000, 1350000, 1400000,
                     1450000, 1500000, 2000000, 1800000, 1600000]
    test_prediction(holiday_spike, "(Holiday Spike)")
    
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Model Info: {'✅ PASS' if info_ok else '❌ FAIL'}")
    print(f"Example: {'✅ PASS' if example_ok else '❌ FAIL'}")
    print("\n🎯 Tất cả prediction tests đã hoàn thành!")
    print("📖 Xem chi tiết API tại: http://localhost:8000/docs")

def test_batch_predictions():
    """Test batch predictions với nhiều scenarios"""
    print("\n" + "="*60)
    print("🔄 BATCH PREDICTION TESTING")
    print("="*60)
    
    scenarios = {
        "Gradual Increase": [1000000, 1020000, 1040000, 1060000, 1080000, 1100000, 1120000, 1140000, 1160000, 1180000],
        "Steep Increase": [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000],
        "Gradual Decrease": [1500000, 1480000, 1460000, 1440000, 1420000, 1400000, 1380000, 1360000, 1340000, 1320000],
        "Steep Decrease": [2000000, 1900000, 1800000, 1700000, 1600000, 1500000, 1400000, 1300000, 1200000, 1100000],
        "Cyclical": [1000000, 1200000, 1400000, 1600000, 1800000, 1600000, 1400000, 1200000, 1000000, 1200000],
        "Random": [1200000, 1100000, 1300000, 1400000, 1250000, 1350000, 1450000, 1300000, 1400000, 1500000]
    }
    
    results = {}
    
    for scenario_name, sales_history in scenarios.items():
        print(f"\n🔍 Testing: {scenario_name}")
        result = test_prediction(sales_history, f"({scenario_name})")
        if result:
            results[scenario_name] = {
                "predicted": result['predicted_sales'],
                "confidence": result['confidence_score'],
                "trend": "UP" if sales_history[-1] > sales_history[0] else "DOWN" if sales_history[-1] < sales_history[0] else "STABLE"
            }
    
    # Print summary
    print("\n" + "="*60)
    print("📊 BATCH TEST RESULTS")
    print("="*60)
    for scenario, result in results.items():
        print(f"{scenario:20} | ${result['predicted']:>12,.0f} | Confidence: {result['confidence']:.3f} | Trend: {result['trend']}")

if __name__ == "__main__":
    print("🚀 Starting GRU API Tests...")
    print("⚠️  Đảm bảo API đang chạy tại http://localhost:8000")
    print("💡 Chạy: python gru_api.py trước khi test")
    
    # Wait a bit for API to be ready
    time.sleep(2)
    
    # Run tests
    run_all_tests()
    
    # Run batch tests
    test_batch_predictions()
    
    print("\n✅ Tất cả tests hoàn thành!")
    print("🎯 API sẵn sàng để sử dụng!")
