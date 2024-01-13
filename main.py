import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

model = models.resnet50(pretrained=True)
model.eval()


def preprocess_image(image_path):
    """Preprocess the satellite image."""
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)


def predict_crop(image_path):
    """Predict the crop type in the image."""
    processed_image = preprocess_image(image_path)
    with torch.no_grad():
        predictions = model(processed_image)
    return torch.argmax(predictions, dim=1)


# Example
image_path = 'imagine.jpg'
crop_type = predict_crop(image_path)
print("Predicted Crop Type:", crop_type.item())
