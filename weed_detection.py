from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image, ImageDraw
import io
import os

app = FastAPI()

# Check if running on Render or locally
if os.path.exists("/app"):  # Render's default working directory
    model_path = "/app/best.pt"
    output_dir = "/app/returned_weed"  # Render directory
else:
    model_path = "/Users/naren/weed_detection_capstone/yolov5_model/check_backend/best.pt"
    output_dir = "/Users/naren/weed_detection_capstone/returned_weed"  # Local directory

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
model.eval()

# Create the directory to save processed images
os.makedirs(output_dir, exist_ok=True)

@app.post("/detect-weed/")
async def detect_weed(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Perform inference
        results = model(image)

        # Process results and draw bounding boxes
        draw = ImageDraw.Draw(image)
        weed_detected = False
        for *box, confidence, class_id in results.xyxy[0]:  # YOLOv5 format
            if confidence > 0.5:  # Confidence threshold
                x_min, y_min, x_max, y_max = map(int, box)
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
                draw.text((x_min, y_min), "Weed", fill="red")
                weed_detected = True

        # Save the processed image
        output_filename = "weed_detected.jpg"
        output_path = os.path.join(output_dir, output_filename)
        image.save(output_path, format="JPEG")

        # Construct the response
        if weed_detected:
            return {"message": "Weed Detected", "image_path": f"/returned_weed/{output_filename}"}
        else:
            return {"message": "No Weed Detected"}
    except Exception as e:
        return {"message": f"Error processing image: {str(e)}"}
