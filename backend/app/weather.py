from abc import ABC, abstractmethod
from datetime import datetime
import httpx
from .config import settings

class WeatherProvider(ABC):
    @abstractmethod
    async def forecast(self, latitude: float, longitude: float) -> list[dict]: ...

class OpenWeatherProvider(WeatherProvider):
    async def forecast(self, latitude, longitude):
        if not settings.openweather_api_key: return []
        async with httpx.AsyncClient(timeout=15) as client:
            r=await client.get(f"{settings.openweather_base_url}/forecast", params={"lat":latitude,"lon":longitude,"appid":settings.openweather_api_key,"units":"metric"}); r.raise_for_status()
        return [{"time":x["dt_txt"],"temp_c":x["main"]["temp"],"humidity":x["main"]["humidity"],"rainfall_mm":x.get("rain",{}).get("3h",0)} for x in r.json()["list"]]

weather_provider: WeatherProvider = OpenWeatherProvider()
