import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = "models/model.json"
LABELS_PATH = "models/labels.txt"

# NOTE: If your model is in TF.js format, you must first convert it to a Keras/TF SavedModel or H5.
# Example: Use tensorflowjs_converter before deploying.
model = tf.keras.models.load_model("models/my_model")  # <-- update after conversion

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines() if line.strip()]


def preprocess_image(image_bytes):
    """Resize and normalize image for model input."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  # adjust if your model expects another size
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)


@app.post("/api/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)
        predictions = model.predict(tensor)[0]

        max_index = int(np.argmax(predictions))
        breed = labels[max_index] if max_index < len(labels) else "Unknown"
        confidence = float(predictions[max_index])

        return {
            "breed": breed,
            "confidence": confidence,
            "method": "FastAPI + TensorFlow"
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "labels": len(labels)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
