from hashlib import sha256
from pathlib import Path
from twilio.rest import Client
from .config import settings

def send_alert(phone:str, message:str) -> str | None:
    """Send SMS, or a WhatsApp sandbox message when TWILIO_FROM_NUMBER starts with whatsapp:."""
    if not all([phone,settings.twilio_account_sid,settings.twilio_auth_token,settings.twilio_from_number]): return None
    to=phone if phone.startswith("whatsapp:") or not settings.twilio_from_number.startswith("whatsapp:") else f"whatsapp:{phone}"
    return Client(settings.twilio_account_sid,settings.twilio_auth_token).messages.create(to=to,from_=settings.twilio_from_number,body=message).sid

def generate_marathi_audio(text:str) -> str | None:
    if not(settings.elevenlabs_api_key and settings.elevenlabs_voice_id): return None
    from elevenlabs.client import ElevenLabs
    audio=b"".join(ElevenLabs(api_key=settings.elevenlabs_api_key).text_to_speech.convert(voice_id=settings.elevenlabs_voice_id,model_id="eleven_multilingual_v2",text=text))
    name=sha256(text.encode()).hexdigest()+".mp3"; path=Path(settings.audio_dir);path.mkdir(parents=True,exist_ok=True); (path/name).write_bytes(audio)
    return f"/audio/{name}"
