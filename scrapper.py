import requests
import random
from bs4 import BeautifulSoup

key = random.randint(1, 99)
headers = {'user_agent': f'yo{key}'}

# Fetch the webpage
fetch = requests.get("https://pizzagalleria.in/our-presence/", headers=headers)
print("fetch status:", fetch.status_code)

# Parse the HTML content
soup = BeautifulSoup(fetch.content, 'html.parser')

# Extract outlet information
outlets = soup.find_all('div', class_='elementor-image-box-content') + \
          soup.find_all('h3', class_='elementor-image-box-title')

print("Following the outlets of Pizza Galleria:\n")
with open('RAG_business\Info_scrapped.txt', 'w') as file:  # Open file in append mode
    for item in outlets:
        try:
            title = item.find('a').get_text() if 'h3' in item.name else item.find('h3').get_text()
            addr = item.find('p').get_text()
            outlet_data = f'Outlet in {title}, address is: {addr}\n'
            print(outlet_data)
            file.write(outlet_data)  # Write each outlet's data to the file
        except AttributeError:
            continue