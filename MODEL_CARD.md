# KrishiRaksha model card

## Scope

The model is MobileNetV2 transfer learning trained on the PlantVillage image dataset's 38 labelled classes. PlantVillage images are predominantly leaf-closeups with controlled backgrounds; the training pipeline adds background replacement, motion blur, rotations, zoom, brightness variation and flips to reduce this gap. It is an aid for scouting, not a pesticide prescription or laboratory diagnosis.

It does **not** include cotton classes. Cotton disease support is a roadmap item and requires a suitably licensed, field-representative cotton dataset and separate validation.

## Performance reporting

`ml/train.py` writes `classification_report.json`, containing precision, recall and F1 for every class plus macro/weighted averages, from the held-out stratified 15% test split. Do not claim a single accuracy in isolation: inspect weak classes before deployment.

## Risk-rule sources and limits

Threshold citations are recorded directly next to each rule in `backend/app/risk_rules.py`: Cornell Vegetable MD Online (late blight), University of Florida IFAS (bacterial spot), UC IPM (powdery mildew), and CIP late-blight guidance. Rules are weather screening signals, not confirmations; they should be reviewed against local varietal, location, and extension-service recommendations.
