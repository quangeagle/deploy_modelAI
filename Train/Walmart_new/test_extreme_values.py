#!/usr/bin/env python3
"""
Test script cho extreme values và edge cases
Kiểm tra trường hợp: 8 tuần 0, tuần 9: 400k, tuần 10: 1.2M
"""

import requests
import json
import time
import numpy as np

def test_api_health():
    """Kiểm tra API có hoạt động không"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ API is healthy")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_extreme_values_endpoint():
    """Test endpoint /test-extreme-values"""
    try:
        response = requests.get("http://localhost:8000/test-extreme-values")
        if response.status_code == 200:
            data = response.json()
            print("\n📊 EXTREME VALUES TEST CASES:")
            print("="*60)
            
            for case_name, case_data in data["test_cases"].items():
                print(f"\n🔍 {case_name.upper()}:")
                print(f"   • Description: {case_data['description']}")
                print(f"   • Sales history: {case_data['sales_history']}")
                
                if "expected_issues" in case_data:
                    print(f"   • Expected issues:")
                    for issue in case_data["expected_issues"]:
                        print(f"     - {issue}")
                
                if "solutions_applied" in case_data:
                    print(f"   • Solutions applied:")
                    for solution in case_data["solutions_applied"]:
                        print(f"     - {solution}")
            
            print(f"\n📋 HOW TO TEST:")
            for step, desc in data["how_to_test"].items():
                print(f"   • {step}: {desc}")
            
            print(f"\n🔧 EXPECTED FIXES:")
            for fix, desc in data["expected_fixes"].items():
                print(f"   • {fix}: {desc}")
                
            return True
        else:
            print(f"❌ Failed to get test cases: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing extreme values endpoint: {e}")
        return False

def test_user_case_directly():
    """Test trực tiếp trường hợp của user"""
    try:
        print("\n🧪 TESTING USER'S EXTREME CASE DIRECTLY")
        print("="*60)
        
        # Test case của user
        user_sales_history = [0, 0, 0, 0, 0, 0, 0, 0, 400000, 1200000]
        
        print(f"📊 User's sales history: {user_sales_history}")
        print(f"   • Min: {min(user_sales_history):,.0f}")
        print(f"   • Max: {max(user_sales_history):,.0f}")
        print(f"   • Mean: {np.mean(user_sales_history):,.0f}")
        print(f"   • Std: {np.std(user_sales_history):,.0f}")
        
        # Test GRU standalone
        print(f"\n🔍 Testing GRU standalone...")
        payload = {
            "sales_history": user_sales_history
        }
        
        response = requests.post("http://localhost:8000/gru-standalone", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ GRU Standalone Result:")
            print(f"   • Predicted sales: ${result['predicted_sales']:,.2f}")
            print(f"   • Confidence score: {result['confidence_score']:.3f}")
            print(f"   • Trend detected: {result['trend_detected']}")
            print(f"   • Was adjusted: {result['was_adjusted']}")
            print(f"   • Message: {result['message']}")
            
            # Kiểm tra xem prediction có hợp lý không
            mean_val = np.mean(user_sales_history)
            std_val = np.std(user_sales_history)
            reasonable_min = max(0, mean_val - 3 * std_val)
            reasonable_max = mean_val + 3 * std_val
            
            print(f"\n📏 Reasonable range analysis:")
            print(f"   • Mean ± 3*std: [{reasonable_min:,.0f}, {reasonable_max:,.0f}]")
            print(f"   • Prediction: {result['predicted_sales']:,.0f}")
            
            if reasonable_min <= result['predicted_sales'] <= reasonable_max:
                print(f"   ✅ Prediction trong range hợp lý")
            else:
                print(f"   ⚠️  Prediction ngoài range hợp lý")
                
            return result
        else:
            print(f"❌ GRU standalone failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error testing user case directly: {e}")
        return None

def test_comparison_api():
    """Test comparison API với trường hợp của user"""
    try:
        print(f"\n🔄 TESTING COMPARISON API")
        print("="*60)
        
        # Test case của user
        user_sales_history = [0, 0, 0, 0, 0, 0, 0, 0, 400000, 1200000]
        
        # External factors mẫu
        external_factors_current = {
            "Temperature": 25.0,
            "Fuel_Price": 3.50,
            "CPI": 200.0,
            "Unemployment": 5.0,
            "Holiday_Flag": 0,
            "Month": 6,
            "WeekOfYear": 25,
            "Year": 2024,
            "DayOfWeek": 1,
            "Is_Weekend": 0
        }
        external_factors_previous = {
            "Temperature": 24.0,
            "Fuel_Price": 3.45,
            "CPI": 199.0,
            "Unemployment": 5.1
        }
        
        payload = {
            "sales_history": user_sales_history,
            "external_factors_current": external_factors_current,
            "external_factors_previous": external_factors_previous
        }
        
        print(f"📊 Sending comparison request...")
        response = requests.post("http://localhost:8000/compare", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Comparison API Result:")
            
            # GRU Standalone
            gru = result["gru_standalone"]
            print(f"\n🔍 GRU Standalone:")
            print(f"   • Predicted: ${gru['predicted_sales']:,.2f}")
            print(f"   • Confidence: {gru['confidence_score']:.3f}")
            print(f"   • Trend: {gru['trend_detected']}")
            print(f"   • Adjusted: {gru['was_adjusted']}")
            
            # GRU Ensemble
            ensemble = result["gru_ensemble"]
            print(f"\n🚀 GRU Ensemble:")
            print(f"   • Final prediction: ${ensemble['final_prediction']:,.2f}")
            print(f"   • GRU contribution: ${ensemble['gru_prediction']:,.2f}")
            print(f"   • XGBoost adjustment: {ensemble['xgboost_adjustment_ratio']*100:.2f}%")
            print(f"   • Confidence: {ensemble['confidence_score']:.3f}")
            
            # Comparison Analysis
            comparison = result["comparison_analysis"]
            print(f"\n📊 Comparison Analysis:")
            print(f"   • Absolute difference: ${comparison['prediction_comparison']['absolute_difference']:,.2f}")
            print(f"   • Relative difference: {comparison['prediction_comparison']['relative_difference_percent']:.2f}%")
            print(f"   • Recommendation: {comparison['recommendation']}")
            
            return result
        else:
            print(f"❌ Comparison API failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error testing comparison API: {e}")
        return None

def test_user_case_endpoint():
    """Test endpoint /test-user-case"""
    try:
        print(f"\n🧪 TESTING USER CASE ENDPOINT")
        print("="*60)
        
        response = requests.get("http://localhost:8000/test-user-case")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ User Case Test Result:")
            
            # Test case info
            test_case = result["test_case"]
            print(f"\n📊 Test Case:")
            print(f"   • Description: {test_case['description']}")
            print(f"   • Sales history: {test_case['sales_history']}")
            
            # Range analysis
            range_analysis = test_case["range_analysis"]
            print(f"\n📏 Range Analysis:")
            print(f"   • Min: {range_analysis['min']:,.0f}")
            print(f"   • Max: {range_analysis['max']:,.0f}")
            print(f"   • Mean: {range_analysis['mean']:,.0f}")
            print(f"   • Std: {range_analysis['std']:,.0f}")
            print(f"   • CV: {range_analysis['coefficient_of_variation']:.2f}")
            print(f"   • Zero count: {range_analysis['zero_count']}")
            print(f"   • Non-zero values: {range_analysis['non_zero_values']}")
            
            # GRU result
            gru_result = result["gru_standalone_result"]
            print(f"\n🔍 GRU Standalone Result:")
            print(f"   • Predicted: ${gru_result['predicted_sales']:,.2f}")
            print(f"   • Confidence: {gru_result['confidence_score']:.3f}")
            print(f"   • Trend: {gru_result['trend_detected']}")
            print(f"   • Adjusted: {gru_result['was_adjusted']}")
            print(f"   • Message: {gru_result['message']}")
            
            # Fixes applied
            fixes = result["fixes_applied"]
            print(f"\n🔧 Fixes Applied:")
            for fix, desc in fixes.items():
                print(f"   • {fix}: {desc}")
                
            return result
        else:
            print(f"❌ User case endpoint failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error testing user case endpoint: {e}")
        return None

def main():
    """Main test function"""
    print("🧪 EXTREME VALUES TESTING SUITE")
    print("="*60)
    print("Testing trường hợp: 8 tuần 0, tuần 9: 400k, tuần 10: 1.2M")
    print("="*60)
    
    # Wait for API to start
    print("⏳ Waiting for API to start...")
    time.sleep(2)
    
    # Test 1: Health check
    print("\n1️⃣ Testing API health...")
    if not test_api_health():
        print("❌ API not ready. Please start the API first.")
        return
    
    # Test 2: Extreme values endpoint
    print("\n2️⃣ Testing extreme values endpoint...")
    test_extreme_values_endpoint()
    
    # Test 3: User case endpoint
    print("\n3️⃣ Testing user case endpoint...")
    test_user_case_endpoint()
    
    # Test 4: Direct GRU test
    print("\n4️⃣ Testing GRU standalone directly...")
    test_user_case_directly()
    
    # Test 5: Comparison API
    print("\n5️⃣ Testing comparison API...")
    test_comparison_api()
    
    print(f"\n✅ ALL TESTS COMPLETED!")
    print(f"📊 Check results above để xem cách hệ thống xử lý extreme values")
    print(f"🔧 Các fix đã được áp dụng:")
    print(f"   • Zero value handling")
    print(f"   • Extreme range detection")
    print(f"   • Range protection")
    print(f"   • Confidence adjustment")

if __name__ == "__main__":
    main()
