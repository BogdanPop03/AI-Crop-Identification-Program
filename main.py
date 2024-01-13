import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

try:
    model = models.resnet50(pretrained=True)
    model.eval()
except Exception as e:
    print("Error loading the pre-trained model:", str(e))
    exit()


def preprocess_image(image_path):
    """Preprocess the satellite image."""
    try:
        img = Image.open(image_path)
    except Exception as e:
        print("Error opening the image:", str(e))
        return None

    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])
        return transform(img).unsqueeze(0)
    except Exception as e:
        print("Error applying image transformation:", str(e))
        return None


def predict_crop(image_path):
    """Predict the crop type in the image."""
    processed_image = preprocess_image(image_path)

    if processed_image is None:
        return None

    try:
        with torch.no_grad():
            predictions = model(processed_image)
        return torch.argmax(predictions, dim=1)
    except Exception as e:
        print("Error making predictions:", str(e))
        return None


# Example
image_path = 'imagine.jpg'
crop_type = predict_crop(image_path)

if crop_type is not None:
    print("Predicted Crop Type:", crop_type.item())
else:
    print("Error predicting crop type. Check the logs for details.")
