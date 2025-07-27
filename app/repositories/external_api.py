import requests
from dotenv import load_dotenv
import os
load_dotenv()

class ExternalAPI:

    @staticmethod
    def fetch_data():
        url = os.getenv("URL_API_GATEWAY")

        token = os.getenv("TOKEN_API")
        headers = {
            "Authorization": f"Bearer {token}" if token else ""
        }

        response = requests.get(url, headers=headers)
        return response.json() if response.status_code == 200 else []
