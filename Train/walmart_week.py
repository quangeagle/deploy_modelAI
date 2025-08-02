import pandas as pd

# Đọc dữ liệu (chỉnh lại path nếu cần)
df = pd.read_csv('../Walmart_Sales.csv')

# Sửa lỗi định dạng ngày khác nhau (ví dụ: 19-02-2010 vs 5/2/2010)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')

# Sort theo Store và Date để đảm bảo đúng thứ tự
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

# Tạo các đặc trưng thời gian
df["WeekOfYear"] = df["Date"].dt.isocalendar().week
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

# Tạo Week_Index riêng cho từng store (reset về 0 mỗi store)
df["Week_Index"] = df.groupby("Store").cumcount()

# Sắp xếp lại cột
df = df[["Store", "Date", "Weekly_Sales", "Holiday_Flag","Temperature", "Fuel_Price", "CPI", "Unemployment", "Week_Index", "WeekOfYear", "Month", "Year"]]

# Lưu ra file CSV
df.to_csv("walmart_processed_by_week.csv", index=False)

# In vài dòng đầu
print(df.head(10))
