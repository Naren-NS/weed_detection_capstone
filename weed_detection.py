from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import torch
from PIL import Image, ImageDraw
import io
import os

app = FastAPI()

# Load YOLOv5 model
try:
    model_path = "/app/best.pt"  # Path to your YOLOv5 model in Render
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, trust_repo=True)
    model.eval()  # Set the model to evaluation mode
except Exception as e:
    raise RuntimeError(f"Failed to load YOLOv5 model: {str(e)}")

# Directory to save the detected weed images
output_dir = "/app/returned_weed"
os.makedirs(output_dir, exist_ok=True)

@app.post("/detect-weed/")
async def detect_weed(file: UploadFile = File(...)):
    try:
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
            if confidence > 0.5:  # Confidence threshold
                x_min, y_min, x_max, y_max = map(int, box)
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
                draw.text((x_min, y_min), "Weed", fill="red")  # Add label
                weed_detected = True

        # Save the processed image
        output_path = os.path.join(output_dir, "weed_detected.jpg")
        image.save(output_path, format="JPEG")

        # Return the response
        if weed_detected:
            return {"message": "Weed Detected", "image_path": "/returned_weed/weed_detected.jpg"}
        else:
            return {"message": "No Weed Detected"}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process the image: {str(e)}"}
        )


# Route to serve the processed image
@app.get("/returned_weed/{filename}")
async def get_detected_image(filename: str):
    file_path = os.path.join(output_dir, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Image not found"}
        )
