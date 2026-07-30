import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset import WasteDataset


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model():
    set_seed(42)

    # 1. Jetson GPU Hardware Acceleration Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training will execute on: {device} ---")

    # 2. Paths and Configurations
    PATH_1 = "./data/recycling/images/images"
    PATH_2 = "./data/garbage_v2/original"
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 1e-4

    # 3. Data Transformations (Adding Augmentation for Better Generalization)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. Instantiate Datasets and DataLoaders
    train_dataset = WasteDataset(PATH_1, PATH_2, split='train', transform=train_transforms)
    val_dataset = WasteDataset(PATH_1, PATH_2, split='val', transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = len(train_dataset.classes)
    print(f"Successfully configured {num_classes} distinct waste categories.")
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # 5. Initialize Pre-trained ResNet18 Network
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)

    # 6. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 7. Training Loop with Best-Model Checkpointing
    print("Beginning model training routine...")
    best_val_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions.double() / total_samples
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

        # 8. Validation Evaluation Step
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)

        val_acc = val_correct.double() / val_total
        print(f"Validation Target Accuracy: {val_acc:.4f}\n")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"New best validation accuracy: {best_val_acc:.4f}")

    # 9. Save the Best Trained Weights
    os.makedirs("models", exist_ok=True)
    if best_state is not None:
        torch.save(best_state, "models/recycling_resnet18.pth")
    else:
        torch.save(model.state_dict(), "models/recycling_resnet18.pth")
    print(f"Training finished! Best validation accuracy: {best_val_acc:.4f}")
    print("Model successfully saved to: models/recycling_resnet18.pth")


if __name__ == '__main__':
    train_model()
