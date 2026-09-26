import io
import os

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Model
# --------------------------------------------------

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "model",
    "verisight_resnet18.pth"
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)
model.eval()


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_image(image):

    image = image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)

    predicted_class = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][predicted_class].item()

    labels = {
        0: "AI-Generated",
        1: "Real"
    }

    return {
        "prediction": labels[predicted_class],
        "confidence": round(confidence * 100, 2)
    }