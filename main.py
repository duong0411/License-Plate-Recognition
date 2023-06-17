# Predict detect bằng Python API
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(show=True, source="0")# source="https://nextcity.org/images/made/219951734_2838e034bb_o_840_630_80.jpg")