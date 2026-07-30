import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def predict_image(image_path):
    # 1. Setup hardware acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Define the exact same binary classes
    classes = ['Non-Recyclable', 'Recyclable']
    
    # 3. Define the exact image transformations used during validation
    predict_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 4. Load the image and process it
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image file: {e}")
        return
        
    image_tensor = predict_transforms(image).unsqueeze(0).to(device)
    
    # 5. Recreate the model architecture layout
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(classes))
    
    # 6. Load your trained weights onto the architecture
    model.load_state_dict(torch.load("models/recycling_resnet18.pth", map_location=device))
    model = model.to(device)
    model.eval()
    
    # 7. Run Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        confidence_value = float(torch.clamp(confidence, 0.0, 1.0).item())
        confidence_pct = min(max(confidence_value * 100.0, 0.0), 100.0)
        
    print(f"\n======================================")
    print(f"Image: {image_path}")
    print(f"Prediction: {classes[predicted_idx.item()]}")
    print(f"Confidence: {confidence_pct:.2f}%")
    print(f"======================================\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <path_to_image>")
    else:
        predict_image(sys.argv[1])
