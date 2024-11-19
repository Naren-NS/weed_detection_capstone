# Use the official Python image from the Docker Hub
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the model file explicitly (adjust path if needed)
COPY yolov5_model/check_backend/best.pt /app/best.pt

# Copy the rest of the application code
COPY . /app

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "weed_detection:app", "--host", "0.0.0.0", "--port", "8000"]
