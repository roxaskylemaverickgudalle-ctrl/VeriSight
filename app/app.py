import os

import gradio as gr
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# =========================================================
# Paths and device
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "verisight_resnet18.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# Image preprocessing
# =========================================================

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# Load model
# =========================================================

model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# =========================================================
# Grad-CAM target layer
# =========================================================

target_layers = [
    model.layer4[-1]
]


# =========================================================
# Prediction function
# =========================================================

def predict_image(image):

    if image is None:
        return None, "Please upload an image.", 0.0

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # -------------------------
    # Model prediction
    # -------------------------

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        predicted_class = int(
            torch.argmax(probabilities).item()
        )

        confidence = float(
            probabilities[predicted_class].item()
        )

    class_names = [
        "AI-Generated",
        "Real"
    ]

    label = class_names[predicted_class]

    # -------------------------
    # Grad-CAM
    # -------------------------

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [
        ClassifierOutputTarget(predicted_class)
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    original = np.array(image.resize((224, 224))) / 255.0

    visualization = show_cam_on_image(
        original.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    result_text = (
        f"Prediction: {label}\n"
        f"Confidence: {confidence * 100:.2f}%"
    )

    return (
        visualization,
        result_text,
        confidence
    )


# =========================================================
# Gradio interface
# =========================================================

with gr.Blocks(
    title="VeriSight — AI Image Detection"
) as demo:

    gr.Markdown(
        "# 🔍 VeriSight\n"
        "### AI-Generated Image Detection & Explainability\n\n"
        "Upload an image to classify it as **AI-Generated** "
        "or **Real** and view a Grad-CAM explanation."
    )

    with gr.Row():

        image_input = gr.Image(
            type="pil",
            label="Upload Image"
        )

        explanation_output = gr.Image(
            label="Grad-CAM Explanation"
        )

    prediction_output = gr.Textbox(
        label="Prediction"
    )

    confidence_output = gr.Number(
        label="Confidence",
        precision=4
    )

    analyze_button = gr.Button(
        "🔍 Analyze Image"
    )

    analyze_button.click(
        fn=predict_image,
        inputs=image_input,
        outputs=[
            explanation_output,
            prediction_output,
            confidence_output
        ]
    )


# =========================================================
# Launch
# =========================================================

if __name__ == "__main__":

    demo.launch(
        share=True
    )
