import sys

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image



MODEL_PATH = "models/best_model.pth"
IMAGE_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()



def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(output, dim=1)

        screen_probability = probabilities[0][1].item()

        prediction = 1 if screen_probability >= 0.43 else 0

    return  screen_probability




if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    probability = predict(image_path)

    print(f"{probability:.4f}")