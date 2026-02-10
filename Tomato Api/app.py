from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os
import uuid

app = FastAPI(title="Tomato Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "finaltomato_model.keras")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "predictions")

os.makedirs(UPLOAD_DIR, exist_ok=True)

model = load_model(MODEL_PATH)

CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "Healthy"
]

CONFIDENCE_THRESHOLD = 0.75


def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Generate unique filename
    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Save image
    with open(file_path, "wb") as f:
        f.write(contents)

    # Preprocess for model
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    confidence = float(np.max(preds))
    class_index = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[class_index]

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "prediction": "Unknown",
            "confidence": confidence,
            "image_path": file_path
        }

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "image_path": file_path
    }
