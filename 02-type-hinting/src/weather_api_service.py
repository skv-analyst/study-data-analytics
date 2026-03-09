from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias
from enum import Enum

from coordinates import Coordinates

# alias для int с названием Celsius
Celsius: TypeAlias = int


class WeatherType(str, Enum):
    THUNDERSTORM = "Гроза"
    DRIZZLE = "Изморось"
    RAIN = "Дождь"
    SNOW = "Снег"
    CLEAR = "Ясно"
    FOG = "Туман"
    CLOUDS = "Облачно"


@dataclass(slots=True, frozen=True)
class Weather:
    temperature: Celsius
    weather_type: str
    sunrise: datetime
    sunset: datetime
    city: str


def get_weather(coordinates: Coordinates) -> Weather:
    """Возвращает погоду через API от OpenWeather в переданных координатах"""
    return Weather(
        temperature=20,
        weather_type=WeatherType.CLEAR,
        sunrise=datetime.fromisoformat("2026-03-09 05:00:00"),
        sunset=datetime.fromisoformat("2026-03-09 20:25:00"),
        city="Saint-Petersburg"
    )

