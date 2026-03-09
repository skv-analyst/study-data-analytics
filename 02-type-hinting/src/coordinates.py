from dataclasses import dataclass
from config import settings



# slots - дает быстрый доступ к атрибутам и эффективное хранение в памяти
# frozen - дает неизменяемую структуру как у кортежа
@dataclass(slots=True, frozen=True)
class Coordinates:
    longitude: float
    latitude: float


def get_gps_coordinates() -> Coordinates:
    """Возвращает текущие координаты устройства"""
    return Coordinates(settings.OWM_LATITUDE, settings.OWM_LONGITUDE)


coordinates = get_gps_coordinates()
