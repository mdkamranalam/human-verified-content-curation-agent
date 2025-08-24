import os
import requests
from dotenv import load_dotenv

load_dotenv()
portia_key = os.getenv('PORTIA_API_KEY')
print(f"Portia Key: {portia_key}")
response = requests.get("https://api.portialabs.ai/health", headers={"Authorization": f"Bearer {portia_key}"})
print(f"Status: {response.status_code}")
print(response.text)