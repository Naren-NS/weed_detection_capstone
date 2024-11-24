from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image, ImageDraw
import io
import os

app = FastAPI()

# Path to the locally available YOLOv5 model
model_path = "/app/best.pt"

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
model.eval()  # Set the model to evaluation mode

# Directory to save processed images
output_dir = "/app/returned_weed"
os.makedirs(output_dir, exist_ok=True)

@app.post("/detect-weed/")
async def detect_weed(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Perform inference
        results = model(image)

        # Flag to check if weeds are detected
        weed_detected = False

        # Process results and draw bounding boxes
        draw = ImageDraw.Draw(image)
        for *box, confidence, class_id in results.xyxy[0]:  # YOLOv5 format
            if confidence > 0.5:
                x_min, y_min, x_max, y_max = map(int, box)
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
                draw.text((x_min, y_min), "Weed", fill="red")
                weed_detected = True

        # Save the processed image
        output_path = os.path.join(output_dir, "weed_detected.jpg")
        image.save(output_path, format="JPEG")

        if weed_detected:
            return {"message": "Weed Detected", "image_path": "/returned_weed/weed_detected.jpg"}
        else:
            return {"message": "No Weed Detected"}
    except Exception as e:
        return {"message": f"Error processing image: {str(e)}"}
