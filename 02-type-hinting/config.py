from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OWM_URL: str
    OWM_API_KEY: str
    OWM_LATITUDE: float
    OWM_LONGITUDE: float

    model_config = SettingsConfigDict(
        env_file=".env"
    )
