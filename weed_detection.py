from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
from PIL import Image, ImageDraw
import io
import os

app = FastAPI()

# Load YOLOv5 model using torch.hub with force_reload=True
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/app/best.pt', force_reload=True, trust_repo=True)
model.eval()  # Set the model to evaluation mode

# Directory to save the detected weed images
output_dir = "/app/returned_weed"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

@app.post("/detect-weed/")
async def detect_weed(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Perform inference directly on the image
    results = model(image)

    # Flag to check if any weeds are detected
    weed_detected = False

    # Process the results
    draw = ImageDraw.Draw(image)
    for *box, confidence, class_id in results.xyxy[0]:  # YOLOv5 format
        if confidence > 0.5:  # Adjust threshold as needed
            x_min, y_min, x_max, y_max = map(int, box)
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            draw.text((x_min, y_min), "Weed", fill="red")  # Add label
            weed_detected = True

    # Save the processed image as 'weed_detected.jpg' in the output directory, overwriting each time
    output_path = os.path.join(output_dir, "weed_detected.jpg")
    image.save(output_path, format="JPEG")

    # Return the response
    if weed_detected:
        return {"message": "Weed Detected", "image_path": "/returned_weed/weed_detected.jpg"}
    else:
        return {"message": "No Weed Detected"}
