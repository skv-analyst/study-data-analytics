import requests
from pprint import pprint
from config import Settings

settings = Settings()

response = requests.get(
    url=settings.OWM_URL,
    params={
        "lat": settings.OWM_LATITUDE,
        "lon": settings.OWM_LONGITUDE,
        "appid": settings.OWM_API_KEY,
        "lang": "ru",
        "units":"metric"
    }
)

pprint(response.json())

