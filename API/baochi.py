import requests
from bs4 import BeautifulSoup

url = 'https://vnexpress.net/kinh-doanh'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

for item in soup.select('.title-news a')[:5]:
    print(f"Tiêu đề: {item.text.strip()}\nLink: {item['href']}\n")
