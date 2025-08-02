from pytrends.request import TrendReq
import pandas as pd
import time

pytrends = TrendReq(hl='vi-VN', tz=7)

# Dừng 5 giây trước khi gửi request (tránh Google block)
time.sleep(5)

# Gửi truy vấn
try:
    pytrends.build_payload(kw_list=['nước ngọt', 'bánh kẹo'], geo='VN')
    trend_data = pytrends.interest_over_time()
    
    print(trend_data.tail())

    # Phân tích tuần gần nhất
    trend_data = trend_data.reset_index()
    last_week = trend_data.iloc[-2]
    this_week = trend_data.iloc[-1]

    if this_week['nước ngọt'] > last_week['nước ngọt']:
        print("🔼 Xu hướng tìm kiếm nước ngọt đang tăng.")
    else:
        print("🔽 Xu hướng nước ngọt không tăng.")
except Exception as e:
    print("Lỗi:", e)
