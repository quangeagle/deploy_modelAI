# import numpy as np
# import pandas as pd
# import torch
# import pickle
# from deep_walmart import GRUModel  # Đảm bảo bạn có file model.py chứa GRUModel

# # === Thiết bị tính toán ===
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # === Load mô hình đã lưu ===
# trained_model = GRUModel(input_size=8)
# trained_model.load_state_dict(torch.load("model_checkpoints/best_model.pth", map_location=device))
# trained_model.to(device)
# trained_model.eval()

# # === Load scaler đã lưu ===
# with open("model_checkpoints/input_scaler.pkl", "rb") as f:
#     input_scaler = pickle.load(f)
# with open("model_checkpoints/target_scaler.pkl", "rb") as f:
#     target_scaler = pickle.load(f)

# # === Định nghĩa cột đặc trưng ===
# feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
#                 'CPI', 'Unemployment', 'WeekOfYear', 'Month']
# input_features = ['Holiday_Flag', 'Temperature', 'Fuel_Price',
#                   'CPI', 'Unemployment', 'WeekOfYear', 'Month']
# weekly_sales_col = ['Weekly_Sales']

# # === Dữ liệu baseline cố định ===

# # === Hàm kiểm tra ảnh hưởng của 1 đặc trưng ===
# def test_feature_influence(changing_feature, values_to_test):
#     print(f"\n🔍 Đang kiểm tra ảnh hưởng của '{changing_feature}' đến dự đoán doanh thu:")

#     predictions = []

#     # Dùng dữ liệu gốc đã test thành công làm baseline
#     base_input_df = pd.read_excel("test.xlsx")  # hoặc .read_excel nếu file excel

#     for val in values_to_test:
#         test_df = base_input_df.copy()
#         test_df[changing_feature] = val  # Gán giá trị mới cho đặc trưng cần kiểm tra

#         # Lặp lại xử lý dữ liệu giống như lúc test
#         scaled_inputs_only = input_scaler.transform(test_df[input_features])
#         scaled_weekly_sales = target_scaler.transform(test_df[weekly_sales_col])
#         scaled_input = np.concatenate([scaled_weekly_sales, scaled_inputs_only], axis=1)

#         input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)

#         with torch.no_grad():
#             y_pred_scaled = trained_model(input_tensor).cpu().numpy()
#             y_pred_real = target_scaler.inverse_transform(y_pred_scaled)[0, 0]

#         predictions.append((val, y_pred_real))

#     print(f"\n📊 Kết quả:")
#     for val, pred in predictions:
#         print(f"   {changing_feature} = {val:>7.2f}  ➜  📈 Doanh thu dự đoán: {pred:,.2f} VND")

# # === Thực thi kiểm tra ===
# if __name__ == "__main__":
#     # Các giá trị để kiểm tra
#     temp_values = np.linspace(10, 40, 7)
#     fuel_values = np.linspace(16000, 24000, 5)
#     cpi_values = np.linspace(100, 150, 6)

#     test_feature_influence("Temperature", temp_values)
#     test_feature_influence("Fuel_Price", fuel_values)
#     test_feature_influence("CPI", cpi_values)


import numpy as np
import pandas as pd
import torch
import pickle
from deep_walmart import GRUModel  # Đảm bảo bạn có đúng class GRUModel

# === Thiết bị tính toán ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Load mô hình tốt nhất đã lưu ===
trained_model = GRUModel(input_size=8)
trained_model.load_state_dict(torch.load("model_checkpoints/best_model.pth", map_location=device))
trained_model.to(device)
trained_model.eval()

# === Load scaler đã lưu ===
with open("model_checkpoints/input_scaler.pkl", "rb") as f:
    input_scaler = pickle.load(f)
with open("model_checkpoints/target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

# === Định nghĩa các cột đặc trưng ===
feature_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
                'CPI', 'Unemployment', 'WeekOfYear', 'Month']
input_features = ['Holiday_Flag', 'Temperature', 'Fuel_Price',
                  'CPI', 'Unemployment', 'WeekOfYear', 'Month']
weekly_sales_col = ['Weekly_Sales']

# === Hàm kiểm tra ảnh hưởng của 1 đặc trưng ===
def test_feature_influence(changing_feature, values_to_test):
    print(f"\n🔍 Đang kiểm tra ảnh hưởng của '{changing_feature}' đến dự đoán doanh thu:")

    predictions = []

    # 🧾 Load dữ liệu gốc làm baseline
    base_input_df = pd.read_excel("test.xlsx")  # Nhớ đảm bảo file này có đúng 10 dòng

    for val in values_to_test:
        test_df = base_input_df.copy()
        test_df[changing_feature] = val

        # 🔄 Tiền xử lý như lúc training
        scaled_inputs_only = input_scaler.transform(test_df[input_features])
        scaled_weekly_sales = target_scaler.transform(test_df[weekly_sales_col])
        scaled_input = np.concatenate([scaled_weekly_sales, scaled_inputs_only], axis=1)

        # 📦 Đưa vào mô hình
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred_scaled = trained_model(input_tensor).cpu().numpy()
            y_pred_real = target_scaler.inverse_transform(y_pred_scaled)[0, 0]

        predictions.append((val, y_pred_real))

    # 🖨️ In kết quả
    print(f"\n📊 Kết quả:")
    for val, pred in predictions:
        print(f"   {changing_feature} = {val:>7.2f}  ➜  📈 Doanh thu dự đoán: {pred:,.2f} VND")

# === Chạy kiểm tra ===
if __name__ == "__main__":
    # Các giá trị cần thử nghiệm
    temp_values = np.linspace(10, 40, 7)
    fuel_values = np.linspace(16000, 24000, 5)
    cpi_values = np.linspace(100, 150, 6)

    test_feature_influence("Temperature", temp_values)
    test_feature_influence("Fuel_Price", fuel_values)
    test_feature_influence("CPI", cpi_values)
