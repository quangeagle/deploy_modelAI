# Simple Test Script cho GRU API
# Test nhanh API với các scenarios khác nhau

import requests
import json

# API Configuration
BASE_URL = "http://localhost:8000"

def test_simple_prediction():
    """Test prediction đơn giản"""
    print("🧪 Testing GRU Sales Prediction API")
    print("="*50)
    
    # Test data - Trending up
    sales_history = [1000000, 1050000, 1100000, 1150000, 1200000, 
                     1250000, 1300000, 1350000, 1400000, 1450000]
    
    payload = {
        "sales_history": sales_history
    }
    
    try:
        print("📊 Input Sales History:")
        for i, sales in enumerate(sales_history, 1):
            print(f"   Tuần {i}: ${sales:,.0f}")
        
        print(f"\n🔍 Sending request to {BASE_URL}/predict...")
        
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction Successful!")
            print(f"📈 Predicted Sales: ${result['predicted_sales']:,.2f}")
            print(f"🎯 Confidence Score: {result['confidence_score']:.3f}")
            print(f"💬 Message: {result['message']}")
            
            # Tính trend
            first_week = sales_history[0]
            last_week = sales_history[-1]
            predicted_week = result['predicted_sales']
            
            print(f"\n📊 Trend Analysis:")
            print(f"   First Week: ${first_week:,.0f}")
            print(f"   Last Week: ${last_week:,.0f}")
            print(f"   Predicted: ${predicted_week:,.0f}")
            
            if predicted_week > last_week:
                print("   📈 Trend: UPWARD")
            elif predicted_week < last_week:
                print("   📉 Trend: DOWNWARD")
            else:
                print("   ➡️ Trend: STABLE")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("💡 Đảm bảo API đang chạy: python gru_api.py")

def test_multiple_scenarios():
    """Test nhiều scenarios khác nhau"""
    print("\n" + "="*50)
    print("🔄 TESTING MULTIPLE SCENARIOS")
    print("="*50)
    
    scenarios = {
        "Trending Up": [1000000, 1050000, 1100000, 1150000, 1200000, 1250000, 1300000, 1350000, 1400000, 1450000],
        "Trending Down": [1500000, 1450000, 1400000, 1350000, 1300000, 1250000, 1200000, 1150000, 1100000, 1050000],
        "Stable": [1200000, 1200000, 1200000, 1200000, 1200000, 1200000, 1200000, 1200000, 1200000, 1200000],
        "Volatile": [1000000, 1500000, 800000, 1600000, 900000, 1400000, 1100000, 1300000, 1200000, 1400000]
    }
    
    for scenario_name, sales_history in scenarios.items():
        print(f"\n🔍 Testing: {scenario_name}")
        print(f"   Input: {sales_history[:3]}...{sales_history[-3:]}")
        
        try:
            payload = {"sales_history": sales_history}
            response = requests.post(f"{BASE_URL}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Predicted: ${result['predicted_sales']:,.0f}")
                print(f"   🎯 Confidence: {result['confidence_score']:.3f}")
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_health_check():
    """Test health check"""
    print("\n" + "="*50)
    print("🏥 HEALTH CHECK")
    print("="*50)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ API is Healthy!")
            print(f"   Status: {data['status']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            print(f"   Device: {data['device']}")
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health Check Error: {e}")

if __name__ == "__main__":
    print("🚀 GRU Sales Prediction API - Simple Test")
    print("⚠️  Đảm bảo API đang chạy: python gru_api.py")
    print()
    
    # Test health check first
    test_health_check()
    
    # Test simple prediction
    test_simple_prediction()
    
    # Test multiple scenarios
    test_multiple_scenarios()
    
    print("\n" + "="*50)
    print("✅ Test completed!")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🎯 Ready to use!")
