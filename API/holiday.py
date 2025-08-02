import requests

API_KEY = 'Ui1wZJimj6sRZ2yJ7uUvh88T8o6Hs2iV'
country = 'VN'
year = 2025

url = f'https://calendarific.com/api/v2/holidays?&api_key={API_KEY}&country={country}&year={year}'

response = requests.get(url)
data = response.json()

for holiday in data['response']['holidays']:
    print(f"{holiday['date']['iso']} - {holiday['name']}")
