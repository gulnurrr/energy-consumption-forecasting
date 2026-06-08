import requests
import pandas as pd
import sqlite3
import os
from dotenv import load_dotenv
import json

load_dotenv()

API_KEY = os.getenv('EIA_API_KEY')
db_name = "electricity_data.db"
    
def fetch_api(api_key=API_KEY):
    
    url = f"http://api.eia.gov/v2/electricity/retail-sales/data/?api_key={api_key}&data[]=sales&facets[stateid][]=CO&facets[sectorid][]=RES&frequency=monthly"

    response = requests.get(url)

    if response.status_code == 200:
        print("Data has been fetched")
        return response.json()
    else:
        raise Exception(f"API request failed: {response.status_code}")
    