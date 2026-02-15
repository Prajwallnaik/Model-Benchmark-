import os
import torch
from flask import Flask, request, render_template, redirect, url_for
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

# Load Model
DEVICE = "cpu" # Force CPU for inference to be safe/simple
class_names = ["Tomato_Early_blight", "Tomato_Late_blight"] # Hardcoded based on dataset

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

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            try:
                # Read image
                img_bytes = file.read()
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                
                # Transform
                input_tensor = transform(image).unsqueeze(0).to(DEVICE)
                
                # Predict
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, preds = torch.max(outputs, 1)
                    predicted_class = class_names[preds.item()]
                
                prediction = predicted_class
            except Exception as e:
                print(f"Error during prediction: {e}")
                prediction = "Error processing image"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
