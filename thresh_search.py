import torch
import torch.nn as nn
import numpy as np

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



VAL_DIR = "dataset/val"
MODEL_PATH = "models/best_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")



transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)



model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()



all_labels = []
all_probs = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)

        screen_probs = probs[:, 1]

        all_labels.extend(labels.numpy())
        all_probs.extend(screen_probs.cpu().numpy())



best_threshold = 0.50
best_accuracy = 0

print("\nSearching Threshold...\n")

for threshold in np.arange(0.10, 0.91, 0.01):

    predictions = [
        1 if p >= threshold else 0
        for p in all_probs
    ]

    accuracy = accuracy_score(all_labels, predictions)

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_threshold = threshold

print("=" * 40)
print(f"Best Threshold : {best_threshold:.2f}")
print(f"Validation Accuracy : {best_accuracy*100:.2f}%")
print("=" * 40)


predictions = [
    1 if p >= best_threshold else 0
    for p in all_probs
]

precision = precision_score(all_labels, predictions)
recall = recall_score(all_labels, predictions)
f1 = f1_score(all_labels, predictions)

print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")