from contextlib import asynccontextmanager
import json
from pathlib import Path
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .config import settings
from .schemas import FarmerRegistration
from .storage import repo
from .ml import predictor
from .weather import weather_provider
from .risk_rules import evaluate_risk_rules
from .services import generate_marathi_audio, send_alert

TREATMENTS=json.loads(Path(settings.treatments_path).read_text(encoding="utf-8"))
scheduler=AsyncIOScheduler(timezone="Asia/Kolkata")
def localized_treatment(class_name:str, language:str):
    return TREATMENTS.get(class_name,TREATMENTS["default"])[language]
def local_message(risk:dict, language:str):
    return risk["alertMessage_mr"] if language=="mr" else risk["alertMessage_en"]

async def advisory_for(farmer:dict):
    forecast=await weather_provider.forecast(farmer["latitude"],farmer["longitude"])
    crops=[]
    for crop in farmer["crops"]:
        risks=evaluate_risk_rules(forecast,crop)
        # Every forecast day is displayed so the UI can render a full five-day timeline.
        daily={}
        for p in forecast:
            day=p["time"][:10]; daily.setdefault(day,[]).append(p)
        timeline=[]
        for day,points in daily.items():
            matching=[r for r in risks if r["severity"] in ("high","medium") and r.get("triggeredAt","").startswith(day)]
            level="red" if any(r["severity"]=="high" for r in matching) else "amber" if matching else "green"
            timeline.append({"date":day,"level":level,"reason":local_message(matching[0],farmer["preferredLanguage"]) if matching else "No current weather-rule trigger.","risks":matching})
        crops.append({"crop":crop,"risks":risks,"timeline":timeline})
    return {"farmerId":farmer["farmerId"],"forecast":forecast,"crops":crops}

async def daily_risk_job():
    for farmer in repo.all_farmers():
        try: advisory=await advisory_for(farmer)
        except Exception: continue
        for crop in advisory["crops"]:
            for risk in crop["risks"]:
                if risk["severity"] not in ("medium","high"): continue
                message=local_message(risk,farmer["preferredLanguage"])
                sid=send_alert(farmer["phone"],message)
                repo.event(farmer["farmerId"],"AlertLog",{"crop":crop["crop"],"disease":risk["disease"],"severity":risk["severity"],"message":message,"deliveryId":sid})

@asynccontextmanager
async def lifespan(app):
    scheduler.add_job(daily_risk_job,"cron",hour=6,minute=0,id="daily-weather-risk",replace_existing=True)
    scheduler.start()
    yield
    scheduler.shutdown(wait=False)

app=FastAPI(title="KrishiRaksha API",version="1.0.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=settings.cors_origins.split(","),allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
Path(settings.audio_dir).mkdir(parents=True,exist_ok=True); app.mount("/audio",StaticFiles(directory=settings.audio_dir),name="audio")

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/farmers/register")
def register_farmer(farmer:FarmerRegistration):
    return repo.save_farmer(farmer.model_dump())

@app.post("/predict")
async def predict(farmerId:str=Query(...), image:UploadFile=File(...), voice:bool=False):
    farmer=repo.farmer(farmerId)
    if not farmer: raise HTTPException(404,"Register the farmer before saving a scan.")
    if image.content_type not in ("image/jpeg","image/png","image/webp"): raise HTTPException(415,"Upload a JPEG, PNG, or WebP image.")
    try: result=predictor.predict(await image.read())
    except RuntimeError as error: raise HTTPException(503,str(error))
    language=farmer["preferredLanguage"]; treatment=localized_treatment(result["className"],language)
    result["localized"]={"name":treatment["name"],"language":language};result["treatment"]=treatment
    if voice: result["voiceUrl"]=generate_marathi_audio(f"{treatment['name']}. {treatment['organic']} {treatment['chemical']}") if language=="mr" else None
    result["scanId"]=repo.event(farmerId,"ScanHistory",result)["timestamp"]
    return result

@app.get("/advisory")
async def advisory(farmerId:str):
    farmer=repo.farmer(farmerId)
    if not farmer: raise HTTPException(404,"Farmer not found")
    return await advisory_for(farmer)

@app.get("/scans/history")
def scan_history(farmerId:str): return repo.scans(farmerId)
