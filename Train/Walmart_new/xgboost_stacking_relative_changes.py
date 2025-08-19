import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings
import joblib
from datetime import datetime
import traceback
warnings.filterwarnings('ignore')

# Cấu hình hiển thị
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Thiết lập font cho matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data():
    """
    Load dữ liệu Walmart và GRU predictions cho XGBoost Adjustment model
    """
    print("🔄 Đang load dữ liệu cho XGBoost Adjustment model...")
    
    # Thử đọc file đã merge
    walmart_path = "E:/TrainAI/Train/walmart_processed_by_week_with_gru_pred.csv"
    try:
        df = pd.read_csv(walmart_path)
        print(f"✅ Đã load file: {walmart_path}")
        print(f"📊 Shape: {df.shape}")
        print(f"📅 Date range: {df['Date'].min()} đến {df['Date'].max()}")
        
        # Kiểm tra cột GRU predictions
        if 'gru_pred' in df.columns:
            print(f"✅ Tìm thấy cột GRU predictions")
            print(f"   GRU predictions range: {df['gru_pred'].min():.2f} đến {df['gru_pred'].max():.2f}")
        else:
            print("⚠️ Không tìm thấy cột GRU predictions")
            
    except FileNotFoundError:
        print("⚠️ Không tìm thấy file merged, đang tạo fallback...")
        
        # Fallback: đọc file gốc và tạo dummy GRU predictions
        original_path = "E:/TrainAI/Train/walmart_processed_by_week.csv"
        try:
            df = pd.read_csv(original_path)
            print(f"✅ Đã load file gốc: {original_path}")
            
            # Tạo dummy GRU predictions (naive approach)
            df['gru_pred'] = df['Weekly_Sales'] * 0.95 + np.random.normal(0, 100, len(df))
            print("⚠️ Đã tạo dummy GRU predictions")
        except FileNotFoundError:
            print("❌ Không tìm thấy file nào, vui lòng kiểm tra đường dẫn")
            return None
    
    # Kiểm tra dữ liệu
    print(f"📋 Columns: {list(df.columns)}")
    print(f"🔍 Missing values:\n{df.isnull().sum()}")
    
    # Kiểm tra Weekly_Sales và GRU predictions
    if 'Weekly_Sales' in df.columns and 'gru_pred' in df.columns:
        print(f"🎯 Target adjustment sẽ là: Weekly_Sales - GRU_pred")
        adjustments = df['Weekly_Sales'] - df['gru_pred']
        print(f"   Adjustment range: {adjustments.min():.2f} đến {adjustments.max():.2f}")
        print(f"   Adjustment mean: {adjustments.mean():.2f}")
    
    return df

def prepare_features_absolute(df):
    """
    Chuẩn bị features cho XGBoost Adjustment model: sử dụng giá trị tuyệt đối của tuần dự đoán
    """
    print("🔄 Đang chuẩn bị features cho Adjustment model (Absolute values)...")
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    print("✅ Đã chuẩn bị xong features cho Adjustment model (Absolute values)")
    return df

def prepare_features_relative(df):
    """
    Chuẩn bị features cho XGBoost Adjustment model: sử dụng thay đổi tương đối so với tuần trước
    """
    print("🔄 Đang chuẩn bị features cho Adjustment model (Relative changes)...")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Tính thay đổi tương đối so với tuần trước
    df['Temperature_change'] = df['Temperature'].pct_change() * 100  # %
    df['Fuel_Price_change'] = df['Fuel_Price'].pct_change() * 100   # %
    df['CPI_change'] = df['CPI'].pct_change() * 100                 # %
    df['Unemployment_change'] = df['Unemployment'].pct_change() * 100 # %
    
    # Xử lý NaN values từ pct_change() (hàng đầu tiên)
    print("⚠️ Đang xử lý NaN values từ relative changes...")
    df['Temperature_change'] = df['Temperature_change'].fillna(0)  # Không thay đổi
    df['Fuel_Price_change'] = df['Fuel_Price_change'].fillna(0)    # Không thay đổi
    df['CPI_change'] = df['CPI_change'].fillna(0)                  # Không thay đổi
    df['Unemployment_change'] = df['Unemployment_change'].fillna(0) # Không thay đổi
    
    # Giữ nguyên các features khác
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    print("✅ Đã chuẩn bị xong features cho Adjustment model (Relative changes)")
    return df

def prepare_features_and_target(df, feature_type='absolute'):
    """
    Chuẩn bị features và target cho training
    XGBoost sẽ học cách điều chỉnh GRU predictions theo tỷ lệ, không phải giá trị tuyệt đối
    """
    print(f"🔄 Đang chuẩn bị features và target ({feature_type})...")
    
    # Kiểm tra GRU predictions
    if 'gru_pred' not in df.columns:
        print("❌ Không tìm thấy cột gru_pred")
        return None, None, None
    
    # Lọc ra các hàng có GRU predictions (test set)
    df_with_gru = df.dropna(subset=['gru_pred']).copy()
    print(f"📊 Số mẫu có GRU predictions: {len(df_with_gru)}")
    
    if len(df_with_gru) == 0:
        print("❌ Không có mẫu nào có GRU predictions")
        return None, None, None
    
    # Chuẩn bị features dựa trên feature_type
    if feature_type == 'absolute':
        df_with_gru = prepare_features_absolute(df_with_gru)
    else:  # relative
        df_with_gru = prepare_features_relative(df_with_gru)
    
    # Chọn features cho XGBoost (LOẠI BỎ gru_pred vì sẽ dùng để tính target)
    if feature_type == 'absolute':
        candidate_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 
                           'Holiday_Flag', 'Month', 'WeekOfYear', 'Year', 'DayOfWeek', 'Is_Weekend']
    else:  # relative
        candidate_features = ['Temperature_change', 'Fuel_Price_change', 'CPI_change', 'Unemployment_change',
                           'Holiday_Flag', 'Month', 'WeekOfYear', 'Year', 'DayOfWeek', 'Is_Weekend']
    
    # Kiểm tra features có sẵn
    available_features = [col for col in candidate_features if col in df_with_gru.columns]
    print(f"✅ Features có sẵn: {available_features}")
    
    # Chuẩn bị X (features) và y (target)
    X = df_with_gru[available_features].copy()
    
    # 🚀 TARGET MỚI: ĐIỀU CHỈNH TƯƠNG ĐỐI thay vì tuyệt đối
    # Thay vì: adjustment = Weekly_Sales_thực_tế - GRU_prediction
    # Bây giờ: adjustment_ratio = (Weekly_Sales_thực_tế - GRU_prediction) / GRU_prediction
    # XGBoost sẽ học cách dự đoán adjustment_ratio này
    y = (df_with_gru['Weekly_Sales'] - df_with_gru['gru_pred']) / df_with_gru['gru_pred']
    
    print(f"🎯 Target MỚI: XGBoost học cách dự đoán ADJUSTMENT RATIO")
    print(f"   Adjustment ratio = (Weekly_Sales - GRU_pred) / GRU_pred")
    print(f"   Adjustment ratio range: {y.min():.4f} đến {y.max():.4f}")
    print(f"   Adjustment ratio mean: {y.mean():.4f}")
    print(f"   Ví dụ: 0.05 = tăng 5%, -0.03 = giảm 3%")
    
    # Kiểm tra NaN và infinity
    print("🔍 Kiểm tra dữ liệu...")
    print(f"Target NaN: {y.isnull().sum()}")
    print(f"Target Infinity: {np.isinf(y).sum()}")
    print(f"Features NaN: {X.isnull().sum().sum()}")
    
    # Xử lý NaN và infinity trong target
    if y.isnull().sum() > 0 or np.isinf(y).sum() > 0:
        print("⚠️ Đang xử lý NaN và Infinity trong target...")
        # Thay thế NaN và Infinity bằng 0 (không điều chỉnh)
        y = y.replace([np.inf, -np.inf], 0).fillna(0)
        print(f"   Sau khi xử lý: NaN: {y.isnull().sum()}, Infinity: {np.isinf(y).sum()}")
    
    # Xử lý NaN trong features
    if X.isnull().sum().sum() > 0:
        print("⚠️ Đang xử lý NaN trong features...")
        X = X.fillna(X.mean())
    
    # Kiểm tra cuối cùng
    if y.isnull().sum() > 0 or X.isnull().sum().sum() > 0:
        print("❌ Vẫn còn NaN values sau khi xử lý")
        return None, None, None
    
    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}")
    print(f"✅ Target range: {y.min():.4f} đến {y.max():.4f}")
    
    return X, y, available_features

def train_xgboost_model(X, y):
    """
    Train XGBoost model để dự đoán ADJUSTMENT cho GRU predictions
    """
    print("🔄 Đang train XGBoost Adjustment model...")
    
    # Split data với shuffle=False để giữ thứ tự thời gian
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False, random_state=42
    )
    
    print(f"📊 Train set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    print(f"🎯 Target: XGBoost học cách dự đoán ADJUSTMENT")
    print(f"   Adjustment range: {y.min():.2f} đến {y.max():.2f}")
    
    # XGBoost parameters - tối ưu hóa để tránh overfitting
    params = {
        'n_estimators': 200,          # Giảm từ 500
        'learning_rate': 0.03,        # Giảm từ 0.05
        'max_depth': 4,               # Giảm từ 5
        'min_child_weight': 3,        # Tăng regularization
        'subsample': 0.8,             # Thêm regularization
        'colsample_bytree': 0.8,      # Thêm regularization
        'reg_alpha': 0.1,             # L1 regularization
        'reg_lambda': 1.0,            # L2 regularization
        'eval_metric': 'rmse',        # Thêm eval_metric
        'random_state': 42
    }
    
    print(f"⚙️ XGBoost parameters: {params}")
    
    # Train model
    model = xgb.XGBRegressor(**params)
    
    # Early stopping để tránh overfitting
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    print("✅ Đã train xong XGBoost Adjustment model")
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_type, df_test=None):
    """
    Đánh giá model và tính ensemble predictions
    XGBoost giờ đây dự đoán ADJUSTMENT RATIO, không phải Weekly_Sales trực tiếp
    """
    print(f"🔄 Đang đánh giá model ({feature_type})...")
    
    # Predictions - XGBoost dự đoán ADJUSTMENT RATIO
    adjustment_ratio_train_pred = model.predict(X_train)
    adjustment_ratio_test_pred = model.predict(X_test)
    
    print("🎯 XGBoost dự đoán ADJUSTMENT RATIO ((Weekly_Sales - GRU_pred) / GRU_pred)")
    print(f"   Adjustment ratio range: {adjustment_ratio_test_pred.min():.4f} đến {adjustment_ratio_test_pred.max():.4f}")
    print(f"   Ví dụ: 0.05 = tăng 5%, -0.03 = giảm 3%")
    
    # Tính ensemble predictions (GRU + XGBoost adjustment)
    print("🔄 Đang tính ensemble predictions (GRU + XGBoost adjustment ratio)...")
    
    # Lấy GRU predictions từ test set
    # Chúng ta cần df_test để truy cập gru_pred
    if df_test is not None and 'gru_pred' in df_test.columns:
        gru_pred_test = df_test['gru_pred'].iloc[X_test.index]
        print(f"✅ Đã lấy GRU predictions từ df_test")
    else:
        print("⚠️ Không có df_test, sử dụng fallback logic")
        # Fallback: giả sử y_test là adjustment ratio và tính ngược lại
        # Điều này không lý tưởng nhưng để tránh lỗi
        gru_pred_test = np.ones_like(adjustment_ratio_test_pred) * 1000000  # Placeholder 1M
    
    # 🚀 ENSEMBLE MỚI: final_pred = GRU_pred * (1 + adjustment_ratio)
    # Thay vì: final_pred = GRU_pred + adjustment
    # Bây giờ: final_pred = GRU_pred * (1 + adjustment_ratio)
    ensemble_test_pred = gru_pred_test * (1 + adjustment_ratio_test_pred)
    
    print(f"📊 Ensemble MỚI: GRU_pred * (1 + adjustment_ratio)")
    print(f"   GRU predictions range: {gru_pred_test.min():.2f} đến {gru_pred_test.max():.2f}")
    print(f"   XGBoost adjustment ratios range: {adjustment_ratio_test_pred.min():.4f} đến {adjustment_ratio_test_pred.max():.4f}")
    print(f"   Final ensemble range: {ensemble_test_pred.min():.2f} đến {ensemble_test_pred.max():.2f}")
    
    # Tính metrics cho từng model
    print("\n📊 KẾT QUẢ ĐÁNH GIÁ:")
    print("=" * 50)
    if df_test is not None and 'Weekly_Sales' in df_test.columns:
        weekly_sales_test = df_test['Weekly_Sales'].iloc[X_test.index]
        print(f"✅ Đã lấy Weekly_Sales thực tế từ df_test")
    else:
        print("⚠️ Không có df_test, sử dụng fallback logic")
        # Fallback: tính ngược lại từ adjustment ratio
        weekly_sales_test = gru_pred_test * (1 + y_test)
    
    gru_r2 = r2_score(weekly_sales_test, gru_pred_test)
    gru_rmse = np.sqrt(mean_squared_error(weekly_sales_test, gru_pred_test))
    gru_mae = mean_absolute_error(weekly_sales_test, gru_pred_test)
    
    print(f"🎯 GRU Model (Baseline):")
    print(f"   R² Score: {gru_r2:.4f}")
    print(f"   RMSE: {gru_rmse:.2f}")
    print(f"   MAE: {gru_mae:.2f}")
    
    # 2. XGBoost model - dự đoán ADJUSTMENT RATIO
    xgb_r2 = r2_score(y_test, adjustment_ratio_test_pred)  # y_test là adjustment ratio thực tế
    xgb_rmse = np.sqrt(mean_squared_error(y_test, adjustment_ratio_test_pred))
    xgb_mae = mean_absolute_error(y_test, adjustment_ratio_test_pred)
    
    print(f"\n🌳 XGBoost Model (Adjustment Ratio):")
    print(f"   R² Score: {xgb_r2:.4f}")
    print(f"   RMSE: {xgb_rmse:.4f}")
    print(f"   MAE: {xgb_mae:.4f}")
    
    # 3. Ensemble model: GRU_pred * (1 + XGBoost_adjustment_ratio)
    ensemble_r2 = r2_score(weekly_sales_test, ensemble_test_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(weekly_sales_test, ensemble_test_pred))
    ensemble_mae = mean_absolute_error(weekly_sales_test, ensemble_test_pred)
    
    print(f"\n🚀 Ensemble Model (GRU * (1 + XGBoost Adjustment Ratio)):")
    print(f"   R² Score: {ensemble_r2:.4f}")
    print(f"   RMSE: {ensemble_rmse:.2f}")
    print(f"   MAE: {ensemble_mae:.2f}")
    
    # So sánh hiệu suất
    print(f"\n📈 SO SÁNH HIỆU SUẤT:")
    print(f"   Ensemble vs GRU:")
    print(f"     R² improvement: {ensemble_r2 - gru_r2:.4f}")
    print(f"     RMSE improvement: {gru_rmse - ensemble_rmse:.2f}")
    print(f"     MAE improvement: {gru_mae - ensemble_mae:.2f}")
    
    # Kiểm tra overfitting
    train_r2 = r2_score(y_train, model.predict(X_train))
    r2_diff = train_r2 - xgb_r2
    
    if abs(r2_diff) < 0.1:
        print(f"\n✅ Model ổn định (R² diff: {r2_diff:.4f})")
    else:
        print(f"\n⚠️ Model có thể bị overfitting (R² diff: {r2_diff:.4f})")
    
    return {
        'gru': {'r2': gru_r2, 'rmse': gru_rmse, 'mae': gru_mae},
        'xgb': {'r2': xgb_r2, 'rmse': xgb_rmse, 'mae': xgb_mae},
        'ensemble': {'r2': ensemble_r2, 'rmse': ensemble_rmse, 'mae': ensemble_mae},
        'ensemble_predictions': ensemble_test_pred
    }

def calculate_feature_importance(model, X_test, y_test, available_features, feature_type):
    """
    Tính feature importance với nhiều phương pháp cho XGBoost Adjustment model
    """
    print(f"🔄 Đang tính feature importance cho Adjustment model ({feature_type})...")
    
    # 1. Gain Importance (XGBoost)
    gain_importance = model.feature_importances_
    gain_df = pd.DataFrame({
        'Feature': available_features,
        'Gain_Importance': gain_importance
    }).sort_values('Gain_Importance', ascending=False)
    
    # 2. Permutation Importance
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42
    )
    perm_df = pd.DataFrame({
        'Feature': available_features,
        'Permutation_Importance': perm_importance.importances_mean
    }).sort_values('Permutation_Importance', ascending=False)
    
    # 3. SHAP Values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Tính SHAP importance (mean absolute SHAP values)
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'Feature': available_features,
        'SHAP_Importance': shap_importance
    }).sort_values('SHAP_Importance', ascending=False)
    
    print("✅ Đã tính xong feature importance cho Adjustment model")
    print(f"💡 Features quan trọng nhất để dự đoán ADJUSTMENT:")
    print(f"   Top 3 (Gain): {gain_df.head(3)['Feature'].tolist()}")
    
    return gain_df, perm_df, shap_df, shap_values, explainer

def create_visualizations(gain_df, perm_df, shap_df, shap_values, explainer,
                         X_test, y_test, ensemble_predictions, feature_type, available_features, gru_predictions=None):
    """
    Tạo các biểu đồ visualization
    """
    print(f"🎨 Đang tạo biểu đồ ({feature_type})...")
    
    # Tạo thư mục output
    output_dir = f"output_{feature_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Feature Importance Comparison
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Gain Importance
    plt.subplot(2, 2, 1)
    top_gain = gain_df.head(10)
    plt.barh(range(len(top_gain)), top_gain['Gain_Importance'])
    plt.yticks(range(len(top_gain)), top_gain['Feature'])
    plt.xlabel('Gain Importance')
    plt.title('Top 10 Features - Gain Importance')
    plt.gca().invert_yaxis()
    
    # Subplot 2: Permutation Importance
    plt.subplot(2, 2, 2)
    top_perm = perm_df.head(10)
    plt.barh(range(len(top_perm)), top_perm['Permutation_Importance'])
    plt.yticks(range(len(top_perm)), top_perm['Feature'])
    plt.xlabel('Permutation Importance')
    plt.title('Top 10 Features - Permutation Importance')
    plt.gca().invert_yaxis()
    
    # Subplot 3: SHAP Importance
    plt.subplot(2, 2, 3)
    top_shap = shap_df.head(10)
    plt.barh(range(len(top_shap)), top_shap['SHAP_Importance'])
    plt.yticks(range(len(top_shap)), top_shap['SHAP_Importance'])
    plt.xlabel('SHAP Importance')
    plt.title('Top 10 Features - SHAP Importance')
    plt.gca().invert_yaxis()
    
    # Subplot 4: Feature Importance Comparison
    plt.subplot(2, 2, 4)
    comparison_df = pd.merge(gain_df, shap_df, on='Feature', suffixes=('_Gain', '_SHAP'))
    comparison_df = comparison_df.head(10)
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    plt.bar(x - width/2, comparison_df['Gain_Importance'], width, label='Gain', alpha=0.8)
    plt.bar(x + width/2, comparison_df['SHAP_Importance'], width, label='SHAP', alpha=0.8)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Gain vs SHAP Importance')
    plt.xticks(x, comparison_df['Feature'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/feature_importance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance - {feature_type.title()}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. SHAP Dependence Plots cho top 3 features
    top_features = gain_df.head(3)['Feature'].tolist()
    for i, feature in enumerate(top_features):
        if feature in X_test.columns:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature, shap_values, X_test, show=False)
            plt.title(f'SHAP Dependence Plot - {feature}')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/shap_dependence_{feature}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. SHAP Force Plot
    plt.figure(figsize=(12, 8))
    shap.force_plot(explainer.expected_value, shap_values[0:100], X_test.iloc[0:100], show=False)
    plt.title(f'SHAP Force Plot - {feature_type.title()} (First 100 samples)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_force.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Ensemble vs GRU vs Actual Comparison
    plt.figure(figsize=(15, 10))
    
    # Lấy GRU predictions - sử dụng parameter hoặc fallback
    if gru_predictions is not None:
        gru_pred_test = gru_predictions
        print("✅ Sử dụng GRU predictions từ parameter")
    else:
        print("⚠️ Không có GRU predictions, bỏ qua biểu đồ so sánh")
        return
    
    # Subplot 1: Predictions vs Actual
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, gru_pred_test, alpha=0.6, label='GRU Predictions', color='blue')
    plt.scatter(y_test, ensemble_predictions, alpha=0.6, label='Ensemble Predictions', color='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Weekly Sales')
    plt.ylabel('Predicted Weekly Sales')
    plt.title('Predictions vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Residuals comparison
    plt.subplot(2, 2, 2)
    gru_residuals = y_test - gru_pred_test
    ensemble_residuals = y_test - ensemble_predictions
    
    plt.scatter(gru_pred_test, gru_residuals, alpha=0.6, label='GRU Residuals', color='blue')
    plt.scatter(ensemble_predictions, ensemble_residuals, alpha=0.6, label='Ensemble Residuals', color='red')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Time series comparison (first 50 samples)
    plt.subplot(2, 2, 3)
    sample_size = min(50, len(y_test))
    x_axis = range(sample_size)
    
    plt.plot(x_axis, y_test.iloc[:sample_size], 'k-', label='Actual', linewidth=2)
    plt.plot(x_axis, gru_pred_test.iloc[:sample_size], 'b--', label='GRU', linewidth=1.5)
    plt.plot(x_axis, ensemble_predictions[:sample_size], 'r--', label='Ensemble', linewidth=1.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Weekly Sales')
    plt.title('Time Series Comparison (First 50 samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Performance metrics comparison
    plt.subplot(2, 2, 4)
    models = ['GRU', 'Ensemble']
    r2_scores = [r2_score(y_test, gru_pred_test), r2_score(y_test, ensemble_predictions)]
    rmse_scores = [np.sqrt(mean_squared_error(y_test, gru_pred_test)), 
                   np.sqrt(mean_squared_error(y_test, ensemble_predictions))]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width/2, r2_scores, width, label='R² Score', color='skyblue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, rmse_scores, width, label='RMSE', color='lightcoral', alpha=0.7)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('R² Score', color='skyblue')
    ax2.set_ylabel('RMSE', color='lightcoral')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', color='skyblue')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{height:.0f}', ha='center', va='bottom', color='lightcoral')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ensemble_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Đã tạo xong biểu đồ trong thư mục {output_dir}/")

def save_results(gain_df, perm_df, shap_df, y_test, ensemble_predictions, feature_type):
    """
    Lưu kết quả vào file
    """
    print(f"💾 Đang lưu kết quả ({feature_type})...")
    
    # Tạo thư mục output
    output_dir = f"output_{feature_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu feature importance
    gain_df.to_csv(f"{output_dir}/feature_importance_gain.csv", index=False)
    perm_df.to_csv(f"{output_dir}/feature_importance_permutation.csv", index=False)
    shap_df.to_csv(f"{output_dir}/feature_importance_shap.csv", index=False)
    
    # Lưu predictions - y_test giờ là adjustment ratio target
    predictions_df = pd.DataFrame({
        'adjustment_ratio_target': y_test,  # Target thực tế (Weekly_Sales - GRU_pred) / GRU_pred
        'ensemble_prediction': ensemble_predictions
    })
    predictions_df.to_csv(f"{output_dir}/ensemble_predictions.csv", index=False)
    
    # Lưu thêm thông tin về adjustment ratios
    adjustment_ratio_stats = {
        'Min': y_test.min(),
        'Max': y_test.max(),
        'Mean': y_test.mean(),
        'Std': y_test.std()
    }
    
    # Lưu stats vào file riêng
    stats_df = pd.DataFrame([adjustment_ratio_stats])
    stats_df.to_csv(f"{output_dir}/adjustment_ratio_stats.csv", index=False)
    
    # Lưu adjustment ratio targets
    adjustment_ratio_targets_df = pd.DataFrame({
        'adjustment_ratio_target': y_test
    })
    adjustment_ratio_targets_df.to_csv(f"{output_dir}/adjustment_ratio_targets.csv", index=False)
    
    print(f"✅ Đã lưu kết quả vào thư mục: {output_dir}")
    print(f"💡 Lưu ý: y_test giờ là ADJUSTMENT RATIO target ((Weekly_Sales - GRU_pred) / GRU_pred)")
    print(f"   Ví dụ: 0.05 = tăng 5%, -0.03 = giảm 3%")

def print_summary(gain_df, perm_df, shap_df, feature_type):
    """
    In summary kết quả
    """
    print(f"\n📋 SUMMARY KẾT QUẢ ({feature_type.upper()}):")
    print("=" * 60)
    
    # Top features theo Gain Importance
    print(f"🏆 TOP 10 FEATURES (Gain Importance):")
    for i, (_, row) in enumerate(gain_df.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['Feature']:<20} | Importance: {row['Gain_Importance']:.4f}")
    
    # Top features theo SHAP Importance
    print(f"\n🔍 TOP 10 FEATURES (SHAP Importance):")
    for i, (_, row) in enumerate(shap_df.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['Feature']:<20} | SHAP: {row['SHAP_Importance']:.4f}")
    
    # Top features theo Permutation Importance
    print(f"\n🔄 TOP 10 FEATURES (Permutation Importance):")
    for i, (_, row) in enumerate(perm_df.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['Feature']:<20} | Permutation: {row['Permutation_Importance']:.4f}")
    
    print(f"\n📁 Files đã tạo:")
    print(f"   - feature_importance_gain.csv")
    print(f"   - feature_importance_permutation.csv") 
    print(f"   - feature_importance_shap.csv")
    print(f"   - ensemble_predictions.csv")
    print(f"   - adjustment_ratio_stats.csv")
    print(f"   - adjustment_ratio_targets.csv")
    print(f"   - Các biểu đồ PNG trong thư mục output_{feature_type}/")
    
    print(f"\n💡 LƯU Ý QUAN TRỌNG:")
    print(f"   - Model này dự đoán ADJUSTMENT RATIO cho GRU predictions")
    print(f"   - Target: (Weekly_Sales - GRU_pred) / GRU_pred")
    print(f"   - Final prediction = GRU_pred * (1 + adjustment_ratio)")
    print(f"   - Ví dụ: adjustment_ratio = 0.05 → tăng 5%")
    print(f"   - Ví dụ: adjustment_ratio = -0.03 → giảm 3%")

def export_model(model, feature_columns, feature_type, performance_metrics):
    """
    Xuất model đã train để sử dụng trong API
    """
    print(f"\n💾 ĐANG XUẤT MODEL CHO {feature_type.upper()}...")
    
    # Tạo thư mục output
    output_dir = f'output_{feature_type}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Lưu XGBoost model
    model_path = f'{output_dir}/xgb_model.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Đã lưu XGBoost model tại: {model_path}")
    
    # 2. Lưu feature columns
    feature_path = f'{output_dir}/feature_columns.txt'
    with open(feature_path, 'w') as f:
        for col in feature_columns:
            f.write(f"{col}\n")
    print(f"✅ Đã lưu feature columns tại: {feature_path}")
    
    # 3. Lưu model info
    info_path = f'{output_dir}/model_info.txt'
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"WALMART SALES PREDICTION MODEL INFO\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Ngày xuất: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Loại model: XGBoost Adjustment Ratio cho GRU\n")
        f.write(f"Tỉ lệ ensemble: GRU_pred * (1 + XGBoost_adjustment_ratio)\n")
        f.write(f"Feature type: {feature_type}\n\n")
        
        f.write(f"PERFORMANCE METRICS:\n")
        f.write(f"-" * 30 + "\n")
        f.write(f"GRU R²: {performance_metrics['gru']['r2']:.4f}\n")
        f.write(f"XGBoost Adjustment Ratio R²: {performance_metrics['xgb']['r2']:.4f}\n")
        f.write(f"Ensemble R²: {performance_metrics['ensemble']['r2']:.4f}\n")
        f.write(f"Ensemble RMSE: {performance_metrics['ensemble']['rmse']:.2f}\n")
        f.write(f"Ensemble MAE: {performance_metrics['ensemble']['mae']:.2f}\n\n")
        
        f.write(f"FEATURES ({len(feature_columns)}):\n")
        f.write(f"-" * 30 + "\n")
        for i, col in enumerate(feature_columns, 1):
            f.write(f"{i}. {col}\n")
        f.write(f"\n")
        
        f.write(f"USAGE:\n")
        f.write(f"-" * 30 + "\n")
        f.write(f"1. Load model: joblib.load('{model_path}')\n")
        f.write(f"2. Load features: open('{feature_path}', 'r').read().splitlines()\n")
        f.write(f"3. Predict adjustment ratio: adjustment_ratio = model.predict(features)\n")
        f.write(f"4. Final prediction = GRU_pred * (1 + adjustment_ratio)\n")
        f.write(f"5. Ví dụ: adjustment_ratio = 0.05 → tăng 5%\n")
        f.write(f"6. Ví dụ: adjustment_ratio = -0.03 → giảm 3%\n")
    
    print(f"✅ Đã lưu model info tại: {info_path}")
    
    # 4. Lưu model parameters
    params_path = f'{output_dir}/model_parameters.txt'
    with open(params_path, 'w') as f:
        f.write(f"XGBOOST MODEL PARAMETERS\n")
        f.write(f"=" * 30 + "\n\n")
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
    
    print(f"✅ Đã lưu model parameters tại: {params_path}")
    
    print(f"\n🎉 HOÀN THÀNH XUẤT MODEL!")
    print(f"📁 Model được lưu trong thư mục: {output_dir}")
    print(f"📋 Các file đã tạo:")
    print(f"   - xgb_model.pkl (Model đã train)")
    print(f"   - feature_columns.txt (Danh sách features)")
    print(f"   - model_info.txt (Thông tin chi tiết)")
    print(f"   - model_parameters.txt (Hyperparameters)")
    print(f"\n💡 LƯU Ý QUAN TRỌNG:")
    print(f"   - Model này dự đoán ADJUSTMENT RATIO, không phải giá trị tuyệt đối")
    print(f"   - Final prediction = GRU_pred * (1 + adjustment_ratio)")
    print(f"   - Adjustment ratio sẽ scale theo magnitude của GRU prediction")
    
    return output_dir

def run_analysis(feature_type='absolute'):
    """
    Chạy analysis cho XGBoost Adjustment model với một loại feature cụ thể
    """
    print(f"🔄 Đang chạy analysis cho XGBoost Adjustment model ({feature_type} features)...")
    
    try:
        # Load và prepare data
        df = load_and_prepare_data()
        if df is None:
            return None
            
        # Prepare features và target
        X, y, available_features = prepare_features_and_target(df, feature_type)
        if X is None:
            return None
            
        print(f"📊 Dữ liệu: {len(X)} dòng, {len(X.columns)} features")
        print(f"🎯 Target: XGBoost học cách dự đoán ADJUSTMENT (Weekly_Sales - GRU_pred)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=False, random_state=42
        )
        
        print(f"📚 Train: {len(X_train)} dòng, Test: {len(X_test)} dòng")
        
        # Train XGBoost Adjustment model
        print("🌳 Đang train XGBoost Adjustment model...")
        train_result = train_xgboost_model(X_train, y_train)
        if train_result is None:
            return None
            
        xgb_model, X_train, X_test, y_train, y_test = train_result
            
        # Evaluate model
        print("📊 Đang đánh giá Adjustment model...")
        performance_metrics = evaluate_model(xgb_model, X_train, X_test, y_train, y_test, feature_type, df)
        
        # Calculate feature importance
        print("🔍 Đang tính feature importance cho Adjustment model...")
        gain_df, perm_df, shap_df, shap_values, explainer = calculate_feature_importance(
            xgb_model, X_test, y_test, available_features, feature_type
        )
        
        # Create visualizations
        print("📈 Đang tạo visualizations cho Adjustment model...")
        
        # Lấy GRU predictions từ df để truyền vào create_visualizations
        gru_predictions = None
        if 'gru_pred' in df.columns:
            # Lấy GRU predictions tương ứng với test set
            test_indices = X_test.index
            gru_predictions = df.loc[test_indices, 'gru_pred']
            print(f"✅ Đã lấy GRU predictions cho visualization")
        
        create_visualizations(
            gain_df, perm_df, shap_df, shap_values, explainer,
            X_test, y_test, performance_metrics['ensemble_predictions'], 
            feature_type, available_features, gru_predictions
        )
        
        # Save results
        print("💾 Đang lưu kết quả cho Adjustment model...")
        save_results(gain_df, perm_df, shap_df, y_test, 
                    performance_metrics['ensemble_predictions'], feature_type)
        
        # Print summary
        print_summary(gain_df, perm_df, shap_df, feature_type)
        
        # Export model
        export_model(xgb_model, available_features, feature_type, performance_metrics)
        
        return {
            'performance': performance_metrics,
            'gain_df': gain_df,
            'perm_df': perm_df,
            'shap_df': shap_df
        }
        
    except Exception as e:
        print(f"❌ Lỗi khi chạy analysis cho Adjustment model: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_approaches():
    """
    So sánh cả hai approaches cho XGBoost Adjustment model
    """
    print("🔄 Đang chạy cả hai approaches để so sánh XGBoost Adjustment model...")
    print("=" * 60)
    
    # Chạy analysis cho absolute values
    print("\n1️⃣ PHƯƠNG PHÁP 1: ABSOLUTE VALUES")
    print("-" * 40)
    results_absolute = run_analysis('absolute')
    
    if results_absolute is None:
        print("❌ Không thể chạy absolute values approach")
        return
    
    # Chạy analysis cho relative changes
    print("\n2️⃣ PHƯƠNG PHÁP 2: RELATIVE CHANGES")
    print("-" * 40)
    results_relative = run_analysis('relative')
    
    if results_relative is None:
        print("❌ Không thể chạy relative changes approach")
        return
    
    # So sánh kết quả
    print("\n🏆 SO SÁNH KẾT QUẢ XGBOOST ADJUSTMENT MODEL:")
    print("=" * 60)
    
    print(f"\n🎯 GRU Model (Baseline):")
    print(f"   Absolute R²: {results_absolute['performance']['gru']['r2']:.4f}")
    print(f"   Relative R²: {results_relative['performance']['gru']['r2']:.4f}")
    
    print(f"\n🌳 XGBoost Model (Adjustment):")
    print(f"   Absolute R²: {results_absolute['performance']['xgb']['r2']:.4f}")
    print(f"   Relative R²: {results_relative['performance']['xgb']['r2']:.4f}")
    
    print(f"\n🚀 Ensemble Model (GRU + XGBoost Adjustment):")
    print(f"   Absolute R²: {results_absolute['performance']['ensemble']['r2']:.4f}")
    print(f"   Relative R²: {results_relative['performance']['ensemble']['r2']:.4f}")
    
    print(f"\n📊 RMSE Comparison:")
    print(f"   Absolute: {results_absolute['performance']['ensemble']['rmse']:.2f}")
    print(f"   Relative: {results_relative['performance']['ensemble']['rmse']:.2f}")
    
    # Xác định approach tốt hơn
    if results_absolute['performance']['ensemble']['r2'] > results_relative['performance']['ensemble']['r2']:
        print(f"\n✅ ABSOLUTE VALUES hoạt động tốt hơn cho Adjustment model!")
        print(f"   R² improvement: {results_absolute['performance']['ensemble']['r2'] - results_relative['performance']['ensemble']['r2']:.4f}")
    else:
        print(f"\n✅ RELATIVE CHANGES hoạt động tốt hơn cho Adjustment model!")
        print(f"   R² improvement: {results_relative['performance']['ensemble']['r2'] - results_absolute['performance']['ensemble']['r2']:.4f}")
    
    # So sánh với GRU baseline
    print(f"\n📈 ENSEMBLE vs GRU BASELINE:")
    print(f"   Absolute improvement: {results_absolute['performance']['ensemble']['r2'] - results_absolute['performance']['gru']['r2']:.4f}")
    print(f"   Relative improvement: {results_relative['performance']['ensemble']['r2'] - results_relative['performance']['gru']['r2']:.4f}")
    
    print(f"\n🎉 HOÀN THÀNH SO SÁNH XGBOOST ADJUSTMENT MODEL!")
    print(f"📁 Kết quả được lưu trong:")
    print(f"   - output_absolute/")
    print(f"   - output_relative/")
    print(f"\n💡 Lưu ý: Model này dự đoán ADJUSTMENT, không phải Weekly_Sales trực tiếp")

def main():
    """
    Main function với menu lựa chọn
    """
    print("🚀 WALMART SALES PREDICTION - XGBOOST ADJUSTMENT RATIO MODEL")
    print("=" * 60)
    print("🎯 Mô hình: GRU + XGBoost Adjustment Ratio")
    print("📊 GRU: Dự đoán doanh thu dựa vào lịch sử 10 tuần")
    print("🌳 XGBoost: Điều chỉnh GRU predictions theo tỷ lệ dựa trên external factors")
    print("🔄 Final Prediction = GRU_pred * (1 + adjustment_ratio)")
    print("💡 Adjustment ratio sẽ scale theo magnitude của GRU prediction")
    print("=" * 60)
    
    while True:
        print("\n📋 CHỌN PHƯƠNG PHÁP:")
        print("1. Absolute Values (Giá trị tuyệt đối) - XGBoost Adjustment Ratio")
        print("2. Relative Changes (Thay đổi tương đối) - XGBoost Adjustment Ratio")
        print("3. So sánh cả hai approaches cho XGBoost Adjustment Ratio model")
        print("4. Xuất XGBoost Adjustment Ratio model (Relative Changes)")
        print("5. Thoát")
        
        choice = input("\n👉 Nhập lựa chọn (1-5) cho XGBoost Adjustment Ratio model: ").strip()
        
        if choice == '1':
            print("\n🔄 Đang chạy XGBoost Adjustment Ratio model với Absolute Values approach...")
            run_analysis('absolute')
            break
        elif choice == '2':
            print("\n🔄 Đang chạy XGBoost Adjustment Ratio model với Relative Changes approach...")
            run_analysis('relative')
            break
        elif choice == '3':
            print("\n🔄 Đang chạy cả hai approaches để so sánh XGBoost Adjustment Ratio model...")
            compare_approaches()
            break
        elif choice == '4':
            print("\n🔄 Đang xuất XGBoost Adjustment Ratio model (Relative Changes)...")
            print("💡 Model này sẽ dự đoán ADJUSTMENT RATIO cho GRU predictions")
            print("💡 Final prediction = GRU_pred * (1 + adjustment_ratio)")
            # Chạy analysis và xuất model
            results = run_analysis('relative')
            if results:
                print("✅ Đã xuất XGBoost Adjustment Ratio model thành công!")
                print("📋 Cách sử dụng:")
                print("   1. GRU dự đoán doanh thu cơ bản")
                print("   2. XGBoost dự đoán adjustment ratio dựa trên external factors")
                print("   3. Final = GRU_pred * (1 + adjustment_ratio)")
                print("\n💡 Lưu ý: Model này dự đoán ADJUSTMENT RATIO, không phải Weekly_Sales trực tiếp")
                print("💡 Adjustment ratio sẽ scale theo magnitude của GRU prediction")
            break
        elif choice == '5':
            print("👋 Tạm biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ. Vui lòng chọn 1-5 cho XGBoost Adjustment Ratio model.")

if __name__ == "__main__":
    main()
