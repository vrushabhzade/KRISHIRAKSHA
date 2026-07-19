from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_env: str = "development"
    cors_origins: str = "http://localhost:5173"
    model_path: str = "models/model.tflite"
    model_info_path: str = "models/class_names.json"
    keras_model_path: str = "models/model.h5"
    treatments_path: str = "app/treatments.json"
    audio_dir: str = "generated_audio"
    openweather_api_key: str = ""
    openweather_base_url: str = "https://api.openweathermap.org/data/2.5"
    aws_region: str = "ap-south-1"
    dynamodb_table: str = "KrishiRaksha"
    use_dynamodb: bool = False
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""

settings = Settings()
Path("models").mkdir(exist_ok=True)
