from pytrends.request import TrendReq

pytrends = TrendReq(hl='vi-VN', tz=360)
pytrends.build_payload(['áo sơ mi trắng nữ'], cat=0, timeframe='2023-01-01 2023-12-31', geo='VN', gprop='')

df = pytrends.interest_over_time()
print(df)
