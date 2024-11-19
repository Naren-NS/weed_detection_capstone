from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io

# Initialize the app and load the model
app = FastAPI()
model = torch.hub.load('../model', 'custom', path='best.pt', source='local')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image = Image.open(io.BytesIO(await file.read()))

    # Perform inference
    results = model(image)

    # Process results
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    return {"detections": detections}

# Run the server with: uvicorn main:app --reload

