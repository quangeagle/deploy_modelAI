import requests

API_KEY = 'ad5bab442e3a99d0f2ba01b296bab594'
city = 'Ha Noi'
url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric&lang=vi'

response = requests.get(url)
data = response.json()

print(f"Thời tiết {city} hôm nay: {data['weather'][0]['description']}, nhiệt độ: {data['main']['temp']}°C")
          