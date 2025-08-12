"""
Xuất đúng 599 dòng test (tuần cuối) kèm cột dự đoán GRU `gru_pred` đặt cạnh các cột dữ liệu Walmart.

Cách dùng:
- Sửa 2 biến đường dẫn WALMART_CSV_PATH và GRU_PRED_CSV_PATH bên dưới
- Chạy: python export_gru_test_with_predictions.py

Đầu vào:
- Walmart CSV có các cột như: Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI,
  Unemployment, WeekOfYear, Month, Year ... (có thể có nhiều dòng theo store)
- GRU CSV: có thể chỉ có 1 cột gru_pred (599 dòng) hoặc có [date, gru_pred]

Đầu ra:
- File `gru_test_with_predictions.csv` gồm đúng p dòng (p = số dòng trong GRU CSV), mỗi dòng 1 tuần, 
  gồm các cột dữ liệu Walmart đã gộp theo tuần + cột gru_pred.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# ======= SỬA ĐƯỜNG DẪN TẠI ĐÂY =======
WALMART_CSV_PATH = r"E:\TrainAI\Train\walmart_processed_by_week.csv"
GRU_PRED_CSV_PATH = r"E:\TrainAI\Train\Walmart_new\gru_predictions.csv"  # có thể chỉ có cột gru_pred
OUT_PATH = r"E:\TrainAI\Train\Walmart_new\gru_test_with_predictions.csv"
# ======================================


def ensure_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series)
    except Exception:
        return pd.to_datetime(series, dayfirst=True, errors="coerce")


def load_walmart_weekly(walmart_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(walmart_csv)
    # Chuẩn hóa cột Date
    date_col = None
    for cand in ["date", "Date"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError("Không thấy cột Date/date trong Walmart CSV")
    df["Date"] = ensure_datetime(df[date_col])

    # Chỉ giữ các cột quan trọng nếu có
    keep_candidates: List[str] = [
        "Date",
        "Weekly_Sales",
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "WeekOfYear",
        "Month",
        "Year",
    ]
    keep_cols = [c for c in keep_candidates if c in df.columns]
    df = df[keep_cols].copy()

    # Gộp theo tuần (trong trường hợp có nhiều store):
    agg_rules = {}
    for c in keep_cols:
        if c == "Date":
            continue
        if c == "Holiday_Flag":
            agg_rules[c] = "max"  # có lễ nếu bất kỳ store có lễ
        else:
            agg_rules[c] = "mean"  # trung bình cho numeric

    weekly = df.groupby("Date", as_index=False).agg(agg_rules).sort_values("Date").reset_index(drop=True)
    return weekly


def load_or_map_gru_preds(gru_csv: str | Path, weekly_dates: pd.Series) -> pd.DataFrame:
    gdf = pd.read_csv(gru_csv)
    if "gru_pred" not in gdf.columns:
        raise ValueError("GRU CSV cần có cột 'gru_pred'")

    if "date" in gdf.columns:
        gdf["date"] = ensure_datetime(gdf["date"])
        preds = gdf[["date", "gru_pred"]].copy()
        preds.rename(columns={"date": "Date"}, inplace=True)
        return preds
    else:
        # Map 599 giá trị vào 599 tuần cuối cùng
        p = len(gdf)
        unique_dates = pd.to_datetime(weekly_dates).dropna().drop_duplicates().sort_values().reset_index(drop=True)
        if p > len(unique_dates):
            # Nếu dự đoán nhiều hơn số tuần khả dụng, cắt còn đúng số tuần cuối để khớp
            print(f"[CẢNH BÁO] GRU có {p} dòng > số tuần khả dụng {len(unique_dates)}. Sẽ dùng {len(unique_dates)} giá trị cuối cùng để khớp.")
            gdf = gdf.tail(len(unique_dates)).reset_index(drop=True)
            p = len(gdf)
        mapped_dates = unique_dates.iloc[-p:].reset_index(drop=True)
        preds = pd.DataFrame({
            "Date": mapped_dates,
            "gru_pred": gdf["gru_pred"].values,
        })
        return preds


def main():
    weekly = load_walmart_weekly(WALMART_CSV_PATH)
    preds = load_or_map_gru_preds(GRU_PRED_CSV_PATH, weekly["Date"])

    merged = pd.merge(weekly, preds, on="Date", how="inner").sort_values("Date").reset_index(drop=True)

    # Đảm bảo đúng p dòng bằng cách chọn đúng số dòng cuối tương ứng nếu dư
    p = len(preds)
    if len(merged) > p:
        merged = merged.tail(p).reset_index(drop=True)

    merged.to_csv(OUT_PATH, index=False)
    print(f"✅ Đã lưu {len(merged)} dòng test với gru_pred vào: {OUT_PATH}")


if __name__ == "__main__":
    main()


