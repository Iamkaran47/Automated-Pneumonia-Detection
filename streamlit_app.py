import streamlit as st
from torchvision import transforms
from PIL import Image
import torch
import tempfile
import os


# Load the pre-trained model
model = torch.load('Connection/pneumonia_model.pth')
model.eval()

# Define the transformations for the input data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class_names = ['Bacteria', 'Normal', 'Virus']

# Streamlit app
st.title("X-Ray Image Classification")

# File uploader
image_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    # Create a temporary file for the uploaded image
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(image_file.getvalue())
        temp_file_path = temp_file.name

    image = Image.open(temp_file_path).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)

    # Move the input tensor to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)

    # Perform prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        prediction = class_names[predicted[0].item()]
        confidence_score = confidence[0].item() * 100

    # Display the prediction result
    st.success(f"Prediction: {prediction}")
    st.info(f"Confidence: {confidence_score:.2f}%")

    # Clean up the temporary file
    os.remove(temp_file_path)