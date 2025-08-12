"""
XGBoost stacking với đầu vào dự đoán GRU cho Walmart sales.

Cách dùng:
- Sửa 2 biến đường dẫn CSV bên dưới: WALMART_CSV_PATH và GRU_PRED_CSV_PATH
- Chạy: python xgb_with_gru_stacking.py

Yêu cầu:
- CSV Walmart đã xử lý có các cột:
  ["Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Month", "WeekOfYear", "Year", "Weekly_Sales"]
- CSV dự đoán GRU có các cột: ["date", "gru_pred"]

Kết quả đầu ra (thư mục output cạnh file script):
- feature_importance_gain.csv
- feature_importance_permutation.csv
- shap_summary.png
- shap_force.html
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

# SHAP là tùy chọn: nếu không cài đặt, sẽ bỏ qua phần SHAP và cảnh báo
try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ======= SỬA ĐƯỜNG DẪN TẠI ĐÂY =======
WALMART_CSV_PATH = r"E:\TrainAI\Train\walmart_processed_by_week.csv"
GRU_PRED_CSV_PATH = r"E:\TrainAI\Train\Walmart_new\gru_predictions.csv"  # ví dụ: có cột [date, gru_pred]
# ======================================


def ensure_datetime(series: pd.Series) -> pd.Series:
    """Chuyển cột ngày về datetime, an toàn với nhiều định dạng."""
    try:
        return pd.to_datetime(series)
    except Exception:
        # Thử parse với dayfirst để an toàn hơn
        return pd.to_datetime(series, dayfirst=True, errors="coerce")


def build_date_from_year_week(df: pd.DataFrame, year_col: str = "Year", week_col: str = "WeekOfYear") -> pd.Series:
    """Tạo cột date (thứ Hai của tuần ISO) từ cột năm và tuần.
    Chậm nhưng an toàn cho cỡ dữ liệu vừa.
    """
    def to_monday(row):
        try:
            return pd.Timestamp.fromisocalendar(int(row[year_col]), int(row[week_col]), 1)
        except Exception:
            return pd.NaT

    return df[[year_col, week_col]].apply(to_monday, axis=1)


def read_and_merge(walmart_csv: str | Path, gru_csv: str | Path) -> pd.DataFrame:
    """Đọc hai file CSV và merge theo cột date.
    - Walmart CSV: chứa đặc trưng và Weekly_Sales, có thể có cột Date hoặc (Year, WeekOfYear)
    - GRU CSV: phải có cột date và gru_pred
    """
    walmart_df = pd.read_csv(walmart_csv)

    # Nếu file GRU không tồn tại, tạo fallback dự đoán đơn giản từ doanh thu tuần trước (naive persistence)
    gru_csv_path = Path(gru_csv)
    if not gru_csv_path.exists():
        print("[CẢNH BÁO] Không tìm thấy file GRU dự đoán:", gru_csv_path)
        print("            Sẽ tạo tạm 'gru_pred' = trung bình doanh thu của tuần trước (naive), bạn nên thay bằng file GRU thật.")
        # Chuẩn hóa 'date' cho walmart_df trước khi tạo gru_df tạm
        if "date" in walmart_df.columns:
            walmart_df["date"] = ensure_datetime(walmart_df["date"])
        elif "Date" in walmart_df.columns:
            walmart_df["date"] = ensure_datetime(walmart_df["Date"])
        elif {"Year", "WeekOfYear"}.issubset(walmart_df.columns):
            walmart_df["date"] = build_date_from_year_week(walmart_df, "Year", "WeekOfYear")
        else:
            raise ValueError("Walmart CSV cần có cột 'date' hoặc 'Date' hoặc cặp (Year, WeekOfYear).")

        # Tạo dự đoán mức aggregate theo tuần: trung bình các store cùng ngày rồi shift 1 tuần
        date_avg = walmart_df.groupby("date")["Weekly_Sales"].mean().sort_index()
        gru_series = date_avg.shift(1).bfill()
        gru_df = gru_series.reset_index()
        gru_df.columns = ["date", "gru_pred"]
    else:
        gru_df = pd.read_csv(gru_csv_path)

    # Chuẩn hóa cột ngày ở walmart_df → 'date'
    if "date" in walmart_df.columns:
        walmart_df["date"] = ensure_datetime(walmart_df["date"])
    elif "Date" in walmart_df.columns:
        walmart_df["date"] = ensure_datetime(walmart_df["Date"])
    elif {"Year", "WeekOfYear"}.issubset(walmart_df.columns):
        walmart_df["date"] = build_date_from_year_week(walmart_df, "Year", "WeekOfYear")
    else:
        raise ValueError("Walmart CSV cần có cột 'date' hoặc 'Date' hoặc cặp (Year, WeekOfYear).")

    # Chuẩn hóa cột ngày ở gru_df
    if "gru_pred" not in gru_df.columns:
        raise ValueError("GRU CSV cần có cột 'gru_pred'.")

    if "date" in gru_df.columns:
        gru_df["date"] = ensure_datetime(gru_df["date"])
    else:
        # Nếu không có cột date, tự ánh xạ dự đoán vào các ngày cuối cùng tương ứng
        print("[CẢNH BÁO] GRU CSV không có cột 'date'. Sẽ ghép dự đoán theo các ngày cuối cùng trong Walmart CSV.")
        # Lấy danh sách ngày duy nhất (chuỗi thời gian theo tuần), sắp xếp tăng dần
        unique_dates = walmart_df["date"].dropna().drop_duplicates().sort_values().reset_index(drop=True)
        p = len(gru_df)
        if p > len(unique_dates):
            raise ValueError("Số dòng GRU lớn hơn số ngày trong Walmart CSV. Không thể ghép tự động.")
        mapped_dates = unique_dates.iloc[-p:].reset_index(drop=True)
        tmp = pd.DataFrame({"date": mapped_dates, "gru_pred": gru_df["gru_pred"].values})
        gru_df = tmp

    # Merge theo date (inner để đảm bảo dữ liệu chung)
    merged = pd.merge(walmart_df, gru_df[["date", "gru_pred"]], on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    # Kiểm tra cột bắt buộc
    required_cols = [
        "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "Month", "WeekOfYear", "Year", "Weekly_Sales", "gru_pred"
    ]
    missing = [c for c in required_cols if c not in merged.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc sau khi merge: {missing}")

    # Loại bỏ NA
    merged = merged.dropna(subset=required_cols)
    return merged


def train_test_time_split(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Tạo tập train/test theo thời gian (80/20), giữ nguyên thứ tự (shuffle=False)."""
    num_rows = len(df)
    split_idx = int(num_rows * 0.8)

    X = df[feature_cols]
    y = df[target_col]

    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """Huấn luyện XGBRegressor theo tham số cơ bản yêu cầu."""
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, float, np.ndarray]:
    """Tính R², RMSE, MAE và trả lại y_pred."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    return r2, rmse, mae, y_pred


def xgb_feature_importance_gain(model: XGBRegressor, feature_names: List[str]) -> pd.Series:
    """Lấy gain importance từ booster và map về tên cột thực."""
    booster = model.get_booster()
    gain_score = booster.get_score(importance_type="gain")
    fmap = {f"f{i}": name for i, name in enumerate(feature_names)}
    gain_series = pd.Series({fmap.get(k, k): v for k, v in gain_score.items()})
    # Bổ sung cột không xuất hiện về 0 để có đủ tất cả features
    gain_series = gain_series.reindex(feature_names).fillna(0.0).sort_values(ascending=False)
    return gain_series


def xgb_permutation_importance(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> pd.Series:
    """Tính permutation importance (mean decrease in score)."""
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_series = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
    return perm_series


def save_importance_outputs(gain_series: pd.Series, perm_series: pd.Series, out_dir: Path) -> None:
    """Lưu CSV và biểu đồ PNG cho importance."""
    out_dir.mkdir(parents=True, exist_ok=True)

    gain_df = gain_series.reset_index()
    gain_df.columns = ["feature", "gain_importance"]
    perm_df = perm_series.reset_index()
    perm_df.columns = ["feature", "permutation_importance"]

    gain_csv = out_dir / "feature_importance_gain.csv"
    perm_csv = out_dir / "feature_importance_permutation.csv"
    gain_df.to_csv(gain_csv, index=False)
    perm_df.to_csv(perm_csv, index=False)

    # Vẽ biểu đồ
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, max(3, len(gain_series) * 0.4)))
    gain_series.sort_values().plot(kind="barh", color="#1f77b4")
    plt.title("XGBoost Feature Importance (Gain)")
    plt.xlabel("Gain")
    plt.tight_layout()
    plt.savefig(out_dir / "xgb_feature_importance_gain.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, max(3, len(perm_series) * 0.4)))
    perm_series.sort_values().plot(kind="barh", color="#ff7f0e")
    plt.title("Permutation Importance (XGBoost)")
    plt.xlabel("Mean importance")
    plt.tight_layout()
    plt.savefig(out_dir / "xgb_feature_importance_permutation.png", dpi=300, bbox_inches="tight")
    plt.close()


def shap_outputs(model: XGBRegressor, X_test: pd.DataFrame, out_dir: Path) -> None:
    """Tính SHAP values và lưu summary plot (png) + force plot (html).
    Sẽ bỏ qua nếu không có SHAP.
    """
    if not SHAP_AVAILABLE:
        print("[CẢNH BÁO] Thư viện shap chưa cài đặt. Bỏ qua phần SHAP.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Ưu tiên TreeExplainer cho mô hình cây
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        expected_value = explainer.expected_value
    except Exception:
        # Fallback API mới
        explainer = shap.Explainer(model)
        shap_values_obj = explainer(X_test)
        shap_values = shap_values_obj.values
        expected_value = shap_values_obj.base_values

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Force plot cho 1 quan sát đầu tiên của test
    try:
        force = shap.force_plot(expected_value, shap_values[0, :], X_test.iloc[0, :], matplotlib=False)
        shap.save_html(str(out_dir / "shap_force.html"), force)
    except Exception:
        # Fallback cho dạng object của Explainer mới
        try:
            force = shap.force_plot(expected_value, shap_values[0], X_test.iloc[0, :], matplotlib=False)
            shap.save_html(str(out_dir / "shap_force.html"), force)
        except Exception as e:
            print(f"[SHAP] Không thể lưu force plot: {e}")


def main():
    # 1) Đọc và merge dữ liệu
    merged = read_and_merge(WALMART_CSV_PATH, GRU_PRED_CSV_PATH)

    # 2) Xác định đặc trưng + target
    feature_cols = [
        "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "Month", "WeekOfYear", "Year", "gru_pred"
    ]
    target_col = "Weekly_Sales"

    # 3) Tạo train/test split theo thời gian
    X_train, X_test, y_train, y_test = train_test_time_split(merged, feature_cols, target_col)

    # 4) Huấn luyện XGBoost
    model = train_xgb(X_train, y_train)

    # 5) Đánh giá
    r2, rmse, mae, y_pred = evaluate_model(model, X_test, y_test)

    # 6) Importance
    gain_series = xgb_feature_importance_gain(model, feature_cols)
    perm_series = xgb_permutation_importance(model, X_test, y_test)

    # 7) SHAP
    out_dir = Path(__file__).resolve().parent / "output"
    shap_outputs(model, X_test, out_dir)

    # 8) Lưu importance outputs
    save_importance_outputs(gain_series, perm_series, out_dir)

    # 9) In kết quả
    print("=" * 60)
    print("🏁 Hiệu năng mô hình (Test set)")
    print(f"R²: {r2:.4f} | RMSE: {rmse:,.2f} | MAE: {mae:,.2f}")

    print("\n🔥 Top 10 features (Gain):")
    top10 = gain_series.head(10)
    for name, val in top10.items():
        print(f"- {name}: {val:.6f}")

    print("\n📁 Đã lưu kết quả vào thư mục:", out_dir)
    print("- feature_importance_gain.csv")
    print("- feature_importance_permutation.csv")
    if SHAP_AVAILABLE:
        print("- shap_summary.png")
        print("- shap_force.html")
    else:
        print("(Bỏ qua SHAP do thiếu thư viện shap)")


if __name__ == "__main__":
    main()


