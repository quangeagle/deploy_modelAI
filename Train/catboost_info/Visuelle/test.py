import pandas as pd

df_main = pd.read_csv('store_train_with_image_features.csv')
df_trend = pd.read_csv(r"E:\visuelle2\visuelle2\vis2_gtrends_data.csv")
df_weather = pd.read_csv(r"E:\visuelle2\visuelle2\vis2_weather_data.csv")

print("Main:", df_main.columns)
print("Trend:", df_trend.columns)
print("Weather:", df_weather.columns)
