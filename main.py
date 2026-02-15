import os
import torch
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

# Setup Templates
templates = Jinja2Templates(directory="templates")

# Load Model
DEVICE = "cpu"
class_names = ["Tomato_Early_blight", "Tomato_Late_blight"]

def load_model():
    model = models.mobilenet_v2()
    # Adjust classifier for 2 classes
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
    
    # Load weights
    model_path = os.path.join("artifacts", "MobileNet.pth")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=DEVICE)
        # Filter out 'total_ops' and 'total_params' added by thop
        new_state_dict = {k: v for k, v in state_dict.items() if "total_ops" not in k and "total_params" not in k}
        model.load_state_dict(new_state_dict)
        print("✅ Model loaded successfully.")
    else:
        print("❌ Model file not found!")
    
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    prediction = None
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)
            
            confidence_score = confidence.item()
            predicted_class = class_names[preds.item()]
            
            # Confidence Threshold (e.g., 70%)
            if confidence_score < 0.7:
                prediction = f"Unknown / Not a Tomato Leaf (Confidence: {confidence_score:.2f})"
            else:
                prediction = f"{predicted_class} (Confidence: {confidence_score:.2f})"
    except Exception as e:
        print(f"Error: {e}")
        prediction = "Error processing image"
    
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
