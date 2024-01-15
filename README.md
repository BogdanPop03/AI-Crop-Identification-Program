# AI-Crop-Identification-Program

This Python script uses a pre-trained ResNet-50 model to predict crop types in satellite images located in a specified folder. Here's a breakdown of the code:

1. Importing Libraries:

os: Operating system-specific functionality.
torch: PyTorch, a deep learning library.
transforms from torchvision: For image preprocessing transformations.
Image from PIL: Python Imaging Library, for handling images.
models from torchvision: Pre-trained deep learning models.

2. Loading the Pre-trained Model (models.resnet50):

Tries to load a pre-trained ResNet-50 model from torchvision. If an exception occurs, it prints an error message and exits the program.

3. Image Preprocessing Function (preprocess_image):

Takes an image file path as input.
Attempts to open the image using the PIL library.
Applies a series of image transformations using transforms.Compose:
Resize the image to 256x256 pixels.
Crop the center to 224x224 pixels.
Convert the image to a PyTorch tensor.
Normalize the tensor using specific mean and standard deviation values.
Returns the preprocessed image tensor.

4. Crop Prediction Function (predict_crop):

Takes an image file path as input.
Calls the preprocess_image function to get the preprocessed image tensor.
If the preprocessing fails (returns None), it returns None.
Uses the pre-trained ResNet-50 model to make predictions on the preprocessed image.
Returns the index of the predicted crop type.

5. Processing Images in a Folder (process_images_in_folder):

Takes a folder path as input.
Iterates through all files in the folder with specified image extensions (e.g., '.jpg', '.jpeg', '.png').
Calls the predict_crop function for each image in the folder.
Prints the filename and the predicted crop type if successful, or an error message if the prediction fails.

6. Example Usage (folder_path and process_images_in_folder):

Specifies an example folder path ('images').
Calls the process_images_in_folder function to predict crop types for all images in the specified folder.

In summary, this script provides a simple way to process multiple images in a folder, predict their crop types using a pre-trained ResNet-50 model, and print the results. Users can adapt this code for their specific use cases by changing the folder path or incorporating it into a larger project.
