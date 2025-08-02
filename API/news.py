import requests

API_KEY = 'f655eb67a06f40b792824bc6ff4f54b9'
url = f'https://newsapi.org/v2/everything?q=kinh%20doanh%20Việt%20Nam&apiKey={API_KEY}&language=vi'

response = requests.get(url)
data = response.json()

for article in data['articles'][:5]:
    print(f"Tiêu đề: {article['title']}\nLink: {article['url']}\n")
