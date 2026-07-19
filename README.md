# KrishiRaksha

Crop disease diagnosis with camera/upload, Grad-CAM visualisation, multilingual guidance, weather-driven early warnings, Marathi speech, and farmer scan history.

## Run locally

1. Copy `backend/.env.example` to `backend/.env`. Set `OPENWEATHER_API_KEY`; set AWS/Twilio/ElevenLabs values when enabling those services. `USE_DYNAMODB=false` keeps development data in memory.
2. Start the API: `cd backend; python -m venv .venv; .\.venv\Scripts\activate; pip install -r requirements.txt; uvicorn app.main:app --reload`
3. Start the web app: `cd frontend; npm install; npm run dev`

The API works without cloud credentials in in-memory development mode. A 38-class PlantVillage TFLite model exported from the supplied notebook is installed locally at `backend/models/model.tflite`, with its labels in `backend/models/class_names.json`. This notebook artifact does not include a Keras model, so Grad-CAM is not available for it. To train the project model instead, run `python ml/train.py --dataset C:\path\to\PlantVillage --output backend/models`. This writes `model.h5` (used for Grad-CAM), `model.tflite` (inference), `class_names.json`, and honest per-class `classification_report.json`.

## Deployment

Deploy `frontend` to Vercel (build command `npm run build`, output `dist`) and `backend` to Railway/Render using the included Dockerfile. Configure `VITE_API_URL` with the API URL and backend environment variables in the deployment dashboard.

## DynamoDB single table

`PK=farmerId, SK=FARMER` is the farmer profile. ScanHistory and AlertLog each use `PK=farmerId, SK=ISO-8601 timestamp`, differentiated by `entityType`. This preserves chronological queries per farmer in the `KrishiRaksha` table.

## API

- `POST /farmers/register` — profile, phone, location, crops, language
- `POST /predict?farmerId=…&voice=true` — multipart field `image`
- `GET /advisory?farmerId=…` — forecast-derived crop timeline
- `GET /scans/history?farmerId=…` — reverse chronological scan records

The APScheduler job runs daily at 06:00 Asia/Kolkata and sends only medium/high weather-rule alerts through configured Twilio SMS or WhatsApp sandbox credentials.
