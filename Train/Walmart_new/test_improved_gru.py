# Test Improved GRU Sales Prediction Model
# Test các xu hướng khác nhau với improved model

import requests
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 1. TEST FUNCTIONS ==========
def test_health_check():
    """Test health check endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   • Status: {response.json()['status']}")
            print(f"   • Model loaded: {response.json()['model_loaded']}")
            print(f"   • Device: {response.json()['device']}")
            return True
        else:
            print("❌ Health check failed")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_prediction(sales_history, expected_trend="unknown"):
    """Test prediction với một chuỗi sales history"""
    try:
        payload = {
            "sales_history": sales_history
        }
        
        response = requests.post("http://localhost:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n📊 PREDICTION RESULT:")
            print(f"   • Input trend: {expected_trend}")
            print(f"   • Input sequence: {sales_history[-3:]}...")  # Last 3 values
            print(f"   • Predicted sales: ${result['predicted_sales']:,.2f}")
            print(f"   • Trend detected: {result['trend_detected']}")
            print(f"   • Was adjusted: {result['was_adjusted']}")
            print(f"   • Confidence: {result['confidence_score']:.3f}")
            print(f"   • Message: {result['message']}")
            
            # Validate trend consistency
            last_value = sales_history[-1]
            predicted = result['predicted_sales']
            
            if expected_trend == "decreasing" and predicted > last_value:
                print("   ⚠️  WARNING: Predicted increase for decreasing trend!")
            elif expected_trend == "increasing" and predicted < last_value:
                print("   ⚠️  WARNING: Predicted decrease for increasing trend!")
            else:
                print("   ✅ Trend prediction consistent")
            
            return result
        else:
            print(f"❌ Prediction failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def test_trend_validation():
    """Test trend validation với các xu hướng khác nhau"""
    print("\n" + "="*60)
    print("🧪 TESTING TREND VALIDATION")
    print("="*60)
    
    test_cases = {
        "strong_decreasing": {
            "sales_history": [1450000, 1400000, 1350000, 1300000, 1250000, 
                             1200000, 1150000, 1100000, 1000000, 900000],
            "expected": "should_continue_decreasing"
        },
        "decreasing": {
            "sales_history": [1500000, 1450000, 1400000, 1350000, 1300000, 
                             1250000, 1200000, 1150000, 1100000, 1050000],
            "expected": "should_continue_decreasing"
        },
        "strong_increasing": {
            "sales_history": [1000000, 1100000, 1200000, 1300000, 1400000, 
                             1500000, 1600000, 1700000, 1800000, 1900000],
            "expected": "should_continue_increasing"
        },
        "increasing": {
            "sales_history": [1000000, 1050000, 1100000, 1150000, 1200000, 
                             1250000, 1300000, 1350000, 1400000, 1450000],
            "expected": "should_continue_increasing"
        },
        "stable": {
            "sales_history": [1200000, 1200000, 1200000, 1200000, 1200000, 
                             1200000, 1200000, 1200000, 1200000, 1200000],
            "expected": "should_remain_stable"
        },
        "volatile": {
            "sales_history": [1000000, 1500000, 800000, 1600000, 900000, 
                             1400000, 1100000, 1300000, 1200000, 1400000],
            "expected": "unpredictable"
        }
    }
    
    results = {}
    
    for trend_name, test_case in test_cases.items():
        print(f"\n🔍 Testing {trend_name.upper()} trend...")
        result = test_prediction(test_case["sales_history"], test_case["expected"])
        results[trend_name] = result
    
    return results

def test_model_info():
    """Test model info endpoint"""
    try:
        response = requests.get("http://localhost:8000/model-info")
        if response.status_code == 200:
            info = response.json()
            print(f"\n📋 MODEL INFO:")
            print(f"   • Model type: {info['model_type']}")
            print(f"   • Architecture: {info['architecture']}")
            print(f"   • Features: {info['features']}")
            print(f"   • Trend types: {info['trend_types']}")
            return True
        else:
            print("❌ Model info failed")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_examples():
    """Test examples endpoint"""
    try:
        response = requests.get("http://localhost:8000/example")
        if response.status_code == 200:
            examples = response.json()
            print(f"\n📚 EXAMPLES:")
            for trend, example in examples["examples"].items():
                print(f"   • {trend}: {example['sales_history'][-3:]}...")
            return True
        else:
            print("❌ Examples failed")
            return False
    except Exception as e:
        print(f"❌ Examples error: {e}")
        return False

def test_trend_endpoint():
    """Test /test-trends endpoint"""
    try:
        response = requests.get("http://localhost:8000/test-trends")
        if response.status_code == 200:
            results = response.json()
            print(f"\n🎯 TREND TEST RESULTS:")
            for trend, result in results["trend_tests"].items():
                if "error" not in result:
                    print(f"   • {trend}: ${result['predicted']:,.2f} ({result['trend_detected']})")
                else:
                    print(f"   • {trend}: ERROR - {result['error']}")
            return True
        else:
            print("❌ Trend test failed")
            return False
    except Exception as e:
        print(f"❌ Trend test error: {e}")
        return False

def visualize_trends(results):
    """Visualize kết quả test trends"""
    if not results:
        return
    
    # Prepare data for visualization
    trends = []
    predicted_values = []
    input_values = []
    confidence_scores = []
    
    for trend_name, result in results.items():
        if result and "error" not in result:
            trends.append(trend_name)
            predicted_values.append(result["predicted_sales"])
            input_values.append(result["input_sequence"][-1])  # Last input value
            confidence_scores.append(result["confidence_score"])
    
    if not trends:
        return
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Predicted vs Input values
    ax1.bar(trends, predicted_values, alpha=0.7, label='Predicted')
    ax1.bar(trends, input_values, alpha=0.5, label='Last Input')
    ax1.set_title('Predicted vs Last Input Values')
    ax1.set_ylabel('Sales ($)')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Confidence scores
    ax2.bar(trends, confidence_scores, color='green', alpha=0.7)
    ax2.set_title('Confidence Scores by Trend')
    ax2.set_ylabel('Confidence (0-1)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Trend consistency (predicted direction vs expected)
    trend_consistency = []
    for i, trend in enumerate(trends):
        if "decreasing" in trend.lower():
            expected_direction = -1
        elif "increasing" in trend.lower():
            expected_direction = 1
        else:
            expected_direction = 0
        
        actual_direction = 1 if predicted_values[i] > input_values[i] else -1
        consistency = 1 if expected_direction == actual_direction else 0
        trend_consistency.append(consistency)
    
    ax3.bar(trends, trend_consistency, color=['green' if c else 'red' for c in trend_consistency], alpha=0.7)
    ax3.set_title('Trend Consistency (1=Correct, 0=Incorrect)')
    ax3.set_ylabel('Consistency')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Prediction accuracy (how close to last value)
    accuracy = [abs(p - i) / i * 100 for p, i in zip(predicted_values, input_values)]
    ax4.bar(trends, accuracy, color='orange', alpha=0.7)
    ax4.set_title('Prediction Change % from Last Value')
    ax4.set_ylabel('Change %')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('improved_gru_trend_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved: improved_gru_trend_analysis.png")
    
    # Print summary
    print(f"\n📈 TREND ANALYSIS SUMMARY:")
    print(f"   • Total trends tested: {len(trends)}")
    print(f"   • Average confidence: {np.mean(confidence_scores):.3f}")
    print(f"   • Trend consistency: {np.mean(trend_consistency):.1%}")
    print(f"   • Average change: {np.mean(accuracy):.1f}%")

# ========== 2. MAIN EXECUTION ==========
if __name__ == "__main__":
    print("🧪 IMPROVED GRU SALES PREDICTION TEST")
    print("="*60)
    
    # Wait for API to start
    print("⏳ Waiting for API to start...")
    time.sleep(2)
    
    # Test 1: Health check
    print("\n1️⃣ Testing health check...")
    if not test_health_check():
        print("❌ API not ready. Please start the API first.")
        exit(1)
    
    # Test 2: Model info
    print("\n2️⃣ Testing model info...")
    test_model_info()
    
    # Test 3: Examples
    print("\n3️⃣ Testing examples...")
    test_examples()
    
    # Test 4: Trend validation
    print("\n4️⃣ Testing trend validation...")
    results = test_trend_validation()
    
    # Test 5: Trend endpoint
    print("\n5️⃣ Testing trend endpoint...")
    test_trend_endpoint()
    
    # Test 6: Visualization
    print("\n6️⃣ Creating visualization...")
    visualize_trends(results)
    
    print(f"\n✅ ALL TESTS COMPLETED!")
    print(f"📊 Check improved_gru_trend_analysis.png for detailed analysis")
    
    # Final summary
    print(f"\n📋 FINAL SUMMARY:")
    print(f"   • Improved GRU model với trend validation")
    print(f"   • Enhanced architecture (Bidirectional + Attention)")
    print(f"   • Balanced training với synthetic data")
    print(f"   • Automatic trend detection và adjustment")
    print(f"   • Confidence scoring based on trend consistency")
