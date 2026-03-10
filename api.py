from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import shutil
from ultralytics import YOLO

# Import the prediction function from your script
from inference_eval import process_single_image

app = FastAPI(title="Chess YOLO to FEN API", description="API for predicting chess board FEN from an image")

# Allow CORS for flutter app testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once when the server starts
MODEL_PATH = "runs/manual_annotate_tuning/trial_11/weights/best.pt"
try:
    print(f"Loading YOLO model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

# Create a temporary directory for uploaded images
TEMP_DIR = "api_temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Chess FEN Inference API is running."}

@app.post("/predict/")
async def predict_fen(file: UploadFile = File(...)):
    """
    Endpoint to predict FEN from an uploaded chess board image.
    Flutter app should send a multipart/form-data request with the image file.
    """
    if model is None:
        return JSONResponse(status_code=500, content={"success": False, "error": "Model not loaded on server."})

    try:
        # Generate a unique filename and save the uploaded image temporarily
        file_extension = file.filename.split('.')[-1]
        temp_filename = f"{uuid.uuid4()}.{file_extension}"
        temp_filepath = os.path.join(TEMP_DIR, temp_filename)
        
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run inference (passing output_dir=None skips saving debug/plot images if your script supports it, 
        # otherwise we can just give it a temp dir or keep the default "result/")
        predicted_fen = process_single_image(
            image_path=temp_filepath, 
            model=model, 
            output_dir=None, # Set to None so it doesn't clutter the server with debug images
            grid_method="geometric",
            force_estimate=False
        )
        
        # Clean up the temporary uploaded file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            
        if predicted_fen:
            return JSONResponse(status_code=200, content={"success": True, "fen": predicted_fen})
        else:
            return JSONResponse(status_code=400, content={"success": False, "error": "Could not detect board or generate FEN."})
            
    except Exception as e:
        # Clean up in case of error
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    # Run the server on all interfaces so your phone/emulator can access it
    uvicorn.run(app, host="0.0.0.0", port=8000)
