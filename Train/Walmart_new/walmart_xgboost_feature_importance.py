import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor


def find_dataset_path() -> Path:
    """
    Tìm đường dẫn file walmart_processed_by_week.csv theo thứ tự ưu tiên:
    - Cùng dự án: ../walmart_processed_by_week.csv (từ thư mục Walmart_new)
    - Tuyệt đối: E:\\TrainAI\\Train\\walmart_processed_by_week.csv (môi trường hiện tại)
    - Dò theo Workspace (đi lên 3 cấp rồi ghép đường dẫn tương đối)
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "walmart_processed_by_week.csv",
        Path(r"E:\\TrainAI\\Train\\walmart_processed_by_week.csv"),
        Path(r"E:/TrainAI/Train/walmart_processed_by_week.csv"),
    ]

    # Thử thêm: nếu đang chạy ở môi trường khác, thử lần theo repo root
    repo_root = here.parent.parent if here.name.lower() == "walmart_new" else here
    candidates.append(repo_root / "Train" / "walmart_processed_by_week.csv")

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Không tìm thấy file walmart_processed_by_week.csv. Hãy đặt file tại 'Train/walmart_processed_by_week.csv' hoặc chỉnh sửa đường dẫn trong script."
    )


def select_available_features(df: pd.DataFrame) -> Tuple[List[str], str]:
    """
    Chọn danh sách đặc trưng có trong dataframe cho XGBoost.
    Target: 'Weekly_Sales'.
    """
    target = "Weekly_Sales"

    # Danh sách feature ứng viên (ưu tiên các cột đã dùng trong EDA/mô hình trước)
    candidate_features = [
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Month",
        "WeekOfYear",
        "Year",
        # Không dùng ID nội bộ như Store để chỉ kiểm tra yếu tố bên ngoài
        # Có thể mở rộng thêm nếu dataset có:
        "IsHoliday",
        "Dept",
        "Size",
        "Type",
    ]

    features = [col for col in candidate_features if col in df.columns]
    if not features:
        raise ValueError(
            "Không tìm thấy cột đặc trưng phù hợp. Hãy đảm bảo các cột như Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment, Month, WeekOfYear, Year có trong dữ liệu."
        )

    return features, target


def train_xgb_and_importance(df: pd.DataFrame, features: List[str], target: str, random_state: int = 42):
    # Loại bỏ NA ở các cột cần thiết
    df_clean = df.dropna(subset=features + [target]).copy()

    X_raw = df_clean[features]
    y = df_clean[target]

    # One-hot encode với các cột không phải số
    non_numeric_cols = [c for c in X_raw.columns if not np.issubdtype(X_raw[c].dtype, np.number)]
    if non_numeric_cols:
        X = pd.get_dummies(X_raw, columns=non_numeric_cols, drop_first=False)
    else:
        X = X_raw.copy()

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Mô hình XGBoost Regression cơ bản, đủ mạnh để đánh giá importance
    model = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Đánh giá nhanh
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    # Hỗ trợ sklearn cũ: tự tính RMSE từ MSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("=" * 60)
    print("🏁 Hiệu năng XGBoost (test set)")
    print(f"R²: {r2:.4f} | RMSE: {rmse:,.2f} | MAE: {mae:,.2f}")

    # Feature importance từ mô hình (gain)
    booster = model.get_booster()
    # Map tên cột thành f0, f1 ... (phòng trường hợp booster trả về fN)
    fmap = {f"f{idx}": name for idx, name in enumerate(feature_names)}
    gain_score = booster.get_score(importance_type="gain")
    # Chuyển về tên cột thật
    gain_series = pd.Series({fmap.get(k, k): v for k, v in gain_score.items()})
    gain_series = gain_series.reindex(feature_names).fillna(0.0).sort_values(ascending=False)

    print("\n🔥 Mức độ quan trọng (Gain) từ XGBoost:")
    for name, val in gain_series.items():
        print(f"- {name}: {val:.6f}")

    # Permutation importance (ổn định hơn, tốn thời gian hơn chút)
    print("\n⏳ Tính permutation importance...")
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1)
    perm_series = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)

    print("\n🧪 Permutation importance (mean decrease in score):")
    for name, val in perm_series.items():
        print(f"- {name}: {val:.6f}")

    return model, gain_series, perm_series, (r2, rmse, mae)


def save_importances(gain_series: pd.Series, perm_series: pd.Series, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    gain_df = gain_series.reset_index()
    gain_df.columns = ["feature", "gain_importance"]
    gain_csv = out_dir / "xgb_feature_importance_gain.csv"
    gain_df.to_csv(gain_csv, index=False)

    perm_df = perm_series.reset_index()
    perm_df.columns = ["feature", "permutation_importance"]
    perm_csv = out_dir / "xgb_feature_importance_permutation.csv"
    perm_df.to_csv(perm_csv, index=False)

    print(f"\n💾 Đã lưu CSV importance tại:\n- {gain_csv}\n- {perm_csv}")

    # Vẽ biểu đồ
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, max(3, len(gain_series) * 0.4)))
    gain_series.sort_values().plot(kind="barh", color="#1f77b4")
    plt.title("XGBoost Feature Importance (Gain)")
    plt.xlabel("Gain")
    plt.tight_layout()
    gain_png = out_dir / "xgb_feature_importance_gain.png"
    plt.savefig(gain_png, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, max(3, len(perm_series) * 0.4)))
    perm_series.sort_values().plot(kind="barh", color="#ff7f0e")
    plt.title("Permutation Importance (XGBoost)")
    plt.xlabel("Mean importance")
    plt.tight_layout()
    perm_png = out_dir / "xgb_feature_importance_permutation.png"
    plt.savefig(perm_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📈 Đã lưu biểu đồ tại:\n- {gain_png}\n- {perm_png}")


def main():
    print("=" * 60)
    print("🧠 TÍNH MỨC ĐỘ LIÊN QUAN ĐẶC TRƯNG CHO DOANH THU WALMART (XGBoost)")
    print("=" * 60)

    # 1) Đọc dữ liệu
    csv_path = find_dataset_path()
    print(f"📄 Đang đọc dữ liệu: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2) Bảo đảm Date là datetime & tạo feature thời gian nếu thiếu
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception:
            pass

    if "Month" not in df.columns and "Date" in df.columns:
        df["Month"] = df["Date"].dt.month
    if "Year" not in df.columns and "Date" in df.columns:
        df["Year"] = df["Date"].dt.year
    if "WeekOfYear" not in df.columns and "Date" in df.columns:
        # isocalendar().week cho số tuần trong năm
        try:
            df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
        except Exception:
            pass

    # 3) Chọn feature & target
    features, target = select_available_features(df)
    print("\n🧾 Sử dụng các đặc trưng:")
    print(", ".join(features))
    print(f"🎯 Target: {target}")

    # 4) Train XGBoost và lấy importance
    model, gain_series, perm_series, metrics = train_xgb_and_importance(df, features, target)

    # 5) Lưu kết quả
    out_dir = Path(__file__).resolve().parent
    save_importances(gain_series, perm_series, out_dir)

    print("\n✅ Hoàn tất!")


if __name__ == "__main__":
    main()


