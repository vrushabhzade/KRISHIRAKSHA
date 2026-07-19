"""Weather risk thresholds are screening rules, not a disease diagnosis."""
RISK_RULES = {
 "Tomato": [
  {"disease":"Late blight", "conditions":{"humidity_min":85,"temp_range":[18,24],"consecutive_days":2,"rainfall_mm_min":0}, "alertMessage_mr":"पुढील हवामानात टोमॅटो उशिरा करपण्याचा धोका आहे. पानांची तपासणी करा.", "alertMessage_en":"Weather favours tomato late blight. Inspect leaves and use locally approved protection.", "severity":"high"}, # Cornell Vegetable MD Online, Late blight: cool (18–24°C), wet/high RH conditions.
  {"disease":"Bacterial spot", "conditions":{"humidity_min":70,"temp_range":[25,40],"consecutive_days":1,"rainfall_mm_min":10}, "alertMessage_mr":"जोरदार पाऊस व उष्णतेमुळे बॅक्टेरियल स्पॉटचा धोका आहे.", "alertMessage_en":"Heavy rain and warmth favour bacterial spot; avoid handling wet plants.", "severity":"medium"}, # University of Florida IFAS, bacterial spot spread by wind-driven rain; warm conditions favor disease.
  {"disease":"Powdery mildew", "conditions":{"humidity_max":60,"temp_range":[25,35],"consecutive_days":2,"rainfall_mm_min":0}, "alertMessage_mr":"उष्ण व कोरड्या हवेत भुरीचा धोका आहे. पानांवर पांढरा थर तपासा.", "alertMessage_en":"Warm, dry weather can favour powdery mildew. Check for white growth.", "severity":"medium"} # UC IPM: powdery mildew favoured by warm days and dry foliage; free water inhibits many species.
 ],
 "Potato": [
  {"disease":"Late blight", "conditions":{"humidity_min":85,"temp_range":[18,24],"consecutive_days":2,"rainfall_mm_min":0}, "alertMessage_mr":"बटाट्यात उशिरा करपण्याचा हवामानाधारित धोका आहे.", "alertMessage_en":"Weather favours potato late blight. Scout the crop promptly.", "severity":"high"} # CIP late blight guidance: prolonged leaf wetness/high humidity and 10–25°C promote infection.
 ]
}

def evaluate_risk_rules(forecast_data: list[dict], crop: str) -> list[dict]:
    triggered=[]
    for rule in RISK_RULES.get(crop, []):
        c=rule["conditions"]; run=0
        for point in forecast_data:
            temp=point.get("temp_c", 0); humidity=point.get("humidity", 0); rain=point.get("rainfall_mm", 0)
            humidity_ok=humidity >= c.get("humidity_min", 0) and humidity <= c.get("humidity_max", 100)
            ok=humidity_ok and c["temp_range"][0] <= temp <= c["temp_range"][1] and rain >= c.get("rainfall_mm_min", 0)
            run = run + 1 if ok else 0
            # Forecast has 3-hour buckets; a rule's day count is 8 contiguous buckets/day.
            if run >= c["consecutive_days"] * 8:
                # Keep the first forecast bucket that completes the condition.  The
                # advisory can then show the warning on the relevant date instead
                # of painting the entire five-day timeline as risky.
                triggered.append({
                    **{k:v for k,v in rule.items() if k != "conditions"},
                    "triggeredAt": point.get("time"),
                })
                break
    return triggered
