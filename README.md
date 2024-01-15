# AI-Crop-Identification-Program

This code is a Python script that uses a pre-trained ResNet-50 model from the torchvision library to predict the crop type in a given satellite image. Here's a breakdown of the code:

1. Importing Libraries:

torch: PyTorch, a deep learning library.
transforms from torchvision: For image preprocessing transformations.
Image from PIL: Python Imaging Library, for handling images.
models from torchvision: Pre-trained deep learning models.

2. Loading the Pre-trained Model:

Tries to load a pre-trained ResNet-50 model. If an exception occurs, it prints an error message and exits the program.

3. Image Preprocessing Function (preprocess_image):

Takes an image file path as input.
Attempts to open the image using the PIL library.
Applies a series of image transformations using transforms.Compose.
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

5. Example Usage (image_path and predict_crop):

Specifies an example image file path ('imagine.jpg').
Calls the predict_crop function on the example image.
If the prediction is successful, it prints the predicted crop type. Otherwise, it prints an error message.

In summary, this script demonstrates how to use a pre-trained ResNet-50 model to predict the crop type in a satellite image. The example at the end shows how to use the functions in a practical context. To use this code for another project, you'd need to have PyTorch and torchvision installed, and you can replace the example image path with the path to your own satellite image.
