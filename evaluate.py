import os
import shutil
import torch.nn.functional as F
import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)



TEST_DIR = "dataset/test"
MODEL_PATH = "models/best_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")



test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Test Images: {len(test_dataset)}")
print("Classes:", test_dataset.classes)



model = models.efficientnet_b0(
    weights=None
)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()



def tta_predict(model, images):
    """
    Performs Test-Time Augmentation (TTA) by averaging
    predictions from the original and horizontally flipped images.
    """

    original = images
    flipped = torch.flip(images, dims=[3])   # Flip width

    with torch.no_grad():

        output1 = model(original)
        output2 = model(flipped)

        prob1 = F.softmax(output1, dim=1)
        prob2 = F.softmax(output2, dim=1)

    return (prob1 + prob2) / 2


REAL_AS_SCREEN = "outputs/misclassified/real_as_screen"
SCREEN_AS_REAL = "outputs/misclassified/screen_as_real"

os.makedirs(REAL_AS_SCREEN, exist_ok=True)
os.makedirs(SCREEN_AS_REAL, exist_ok=True)

all_labels = []
all_predictions = []

misclassified = []

with torch.no_grad():

    image_index = 0

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = tta_predict(model, images)

        predictions = torch.argmax(outputs, dim=1)
        #predictions = (probs[:, 1] >= 0.43).long()

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())

        for i in range(len(labels)):

            true_label = labels[i].item()
            pred_label = predictions[i].item()

            if true_label != pred_label:

                image_path, _ = test_dataset.samples[image_index]

                filename = os.path.basename(image_path)

                misclassified.append({
                    "file": filename,
                    "true": test_dataset.classes[true_label],
                    "pred": test_dataset.classes[pred_label]
                })

                if true_label == 0:
                    shutil.copy2(
                        image_path,
                        os.path.join(REAL_AS_SCREEN, filename)
                    )
                else:
                    shutil.copy2(
                        image_path,
                        os.path.join(SCREEN_AS_REAL, filename)
                    )

            image_index += 1



accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)

print("\n========== RESULTS ==========")
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1 Score : {f1*100:.2f}%")

print("\nConfusion Matrix")
print(confusion_matrix(all_labels, all_predictions))

print("\nClassification Report")
print(classification_report(
    all_labels,
    all_predictions,
    target_names=test_dataset.classes
))