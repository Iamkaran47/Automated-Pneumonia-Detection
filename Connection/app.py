# from flask import Flask, request, jsonify, render_template
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import os
# import torch.nn as nn
#
# app = Flask(__name__)
# UPLOAD_FOLDER = "images"
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}
#
#
# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
#
#
# # Load the model
# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# num_ftrs = model.fc.in_features
# num_classes = 3  # Update with the number of classes in your dataset
# model.fc = nn.Linear(num_ftrs, num_classes)
# model.load_state_dict(torch.load('pneumonia_model-state.pth', map_location=torch.device('cpu')))
# model.eval()
#
# # Define the transformations for the input data
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#
# # Class names
# classes = ['bacteria_Pneumonia', 'NORMAL', 'virus_Pneumonia']
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         image_file = request.files['image']
#         if image_file and allowed_file(image_file.filename):
#             image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
#             image_file.save(image_path)
#
#             image = Image.open(image_path)
#             image_tensor = transform(image).unsqueeze(0)
#
#             with torch.no_grad():
#                 output = model(image_tensor)
#                 _, predicted = torch.max(output.data, 1)
#                 predicted_class_name = classes[predicted.item()]
#                 predicted_class_idx = predicted.item()
#                 accuracy_percentage = output.data[0][predicted_class_idx].item() * 100
#
#             return jsonify({'prediction': predicted_class_name, 'accuracy': accuracy_percentage})
#
#     return jsonify({'error': 'Invalid request'})
#
#
# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify, render_template
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import os
# import torch.nn as nn
#
# # Get the absolute path to the project directory
# PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
#
# # Create the 'images' directory if it doesn't exist
# UPLOAD_FOLDER = os.path.join(PROJECT_DIR, 'images')
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
#
# app = Flask(__name__)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}
#
# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
#
# # Load the model
# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# num_ftrs = model.fc.in_features
# num_classes = 3  # Update with the number of classes in your dataset
# model.fc = nn.Linear(num_ftrs, num_classes)
# model.load_state_dict(torch.load('pneumonia_model-state.pth', map_location=torch.device('cpu')))
# model.eval()
#
# # Define the transformations for the input data
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#         inplace=False
#     )
# ])
# # Class names
# classes = ['bacteria_Pneumonia', 'NORMAL', 'virus_Pneumonia']
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file uploaded'}), 400
#
#     image_file = request.files['image']
#
#     if image_file.filename == '':
#         return jsonify({'error': 'No image file selected'}), 400
#
#     if not allowed_file(image_file.filename):
#         return jsonify({'error': 'Invalid file type'}), 400
#
#     image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
#     image_file.save(image_path)
#
#     image = Image.open(image_path)
#     image_tensor = transform(image).unsqueeze(0)
#
#     # Ensure input tensor has the correct shape (3, 224, 224)
#     if image_tensor.shape[0] != 3:
#         # Convert grayscale images to RGB
#         image_tensor = torch.cat([image_tensor] * 3)
#
#     with torch.no_grad():
#         output = model(image_tensor)
#         probabilities = torch.softmax(output, dim=1)  # Apply softmax to get probabilities
#         confidence, predicted = torch.max(probabilities.data, 1)
#         predicted_class_name = classes[predicted.item()]
#         accuracy_percentage = confidence.item() * 100
#
#     return jsonify({'prediction': predicted_class_name, 'accuracy': accuracy_percentage})
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
# 2
# from flask import Flask, request, jsonify, render_template
# import torch
# from torchvision import transforms
# from PIL import Image
# import os
# import torch.nn as nn
#
# # Get the absolute path to the project directory
# PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
#
# # Create the 'images' directory if it doesn't exist
# UPLOAD_FOLDER = os.path.join(PROJECT_DIR, 'images')
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
#
# app = Flask(__name__)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}
#
#
# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
#
#
# # Load the pneumonia detection model
# # model = pneumonia_model.pth()  # Instantiate your trained model here
# # model.load_state_dict(torch.load('pneumonia_model.pth', map_location=torch.device('cpu')))
# # model.eval()
#
# model.eval()
#
# # Define the transformations for the input data
# # Define the transformations for the input data
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#         inplace=False
#     )
# ])
#
#
# # Class names
# classes = ['normal', 'pneumonia']  # Update with appropriate class names for your model
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file uploaded'}), 400
#
#     image_file = request.files['image']
#
#     if image_file.filename == '':
#         return jsonify({'error': 'No image file selected'}), 400
#
#     if not allowed_file(image_file.filename):
#         return jsonify({'error': 'Invalid file type'}), 400
#
#     image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
#     image_file.save(image_path)
#
#     image = Image.open(image_path)
#     image_tensor = transform(image).unsqueeze(0)
#
#     # Ensure input tensor has the correct shape (3, 224, 224)
#     if image_tensor.shape[0] != 3:
#         # Convert grayscale images to RGB
#         image_tensor = torch.cat([image_tensor] * 3)
#
#     with torch.no_grad():
#         output = model(image_tensor)
#         _, predicted = torch.max(output, 1)
#         predicted_class_name = classes[predicted.item()]
#
#     return jsonify({'prediction': predicted_class_name})
#
#
# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify, render_template
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import os
# import torch.nn as nn
#
# # Get the absolute path to the project directory
# PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
#
# # Create the 'images' directory if it doesn't exist
# UPLOAD_FOLDER = os.path.join(PROJECT_DIR, 'images')
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
#
# app = Flask(__name__)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}
#
#
# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
#
#
# # Load the model
# import torch
#
# # Load the entire saved model
# model = torch.load('pneumonia_model.pth')
#
# # Now you can use the `model` for prediction
#
# num_ftrs = model.fc.in_features
# num_classes = 3  # Update with the number of classes in your dataset
# model.fc = nn.Linear(num_ftrs, num_classes)
# model.load_state_dict(torch.load('pneumonia_model.pth', map_location=torch.device('cpu')))
# model.eval()
#
# # Define the transformations for the input data
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#         inplace=False
#     )
# ])
# # Class names
# classes = ['bacteria_Pneumonia', 'NORMAL', 'virus_Pneumonia']
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file uploaded'}), 400
#
#     image_file = request.files['image']
#
#     if image_file.filename == '':
#         return jsonify({'error': 'No image file selected'}), 400
#
#     if not allowed_file(image_file.filename):
#         return jsonify({'error': 'Invalid file type'}), 400
#
#     image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
#     image_file.save(image_path)
#
#     image = Image.open(image_path)
#     image_tensor = transform(image).unsqueeze(0)
#
#     # Ensure input tensor has the correct shape (3, 224, 224)
#     if image_tensor.shape[1] != 3:
#         # Convert grayscale images to RGB
#         image_tensor = torch.cat([image_tensor] * 3)
#
#     with torch.no_grad():
#         output = model(image_tensor)
#         probabilities = torch.softmax(output, dim=1)  # Apply softmax to get probabilities
#         confidence, predicted = torch.max(probabilities.data, 1)
#         predicted_class_name = classes[predicted.item()]
#         accuracy_percentage = confidence.item() * 100
#
#     return jsonify({'prediction': predicted_class_name, 'accuracy': accuracy_percentage})
#
#
# if __name__ == '__main__':
#     app.run(debug=True)

# runned but error
# from flask import Flask, request, jsonify, render_template
# from torchvision import models, transforms
# from PIL import Image
# import torch
# import os
#
# app = Flask(__name__)
# UPLOAD_FOLDER = "images"
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}
#
# # Load the pre-trained model
# model = models.resnet50(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 3)  # Change the final layer to match the number of classes
# model.eval()
#
# # Define the transformations for the input data
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
# ])
#
# classes = ['bacteria_Pneumonia', 'NORMAL', 'virus_Pneumonia']
#
# # Define mean and standard deviation for normalization
# mean = torch.tensor([0.485, 0.456, 0.406])
# std = torch.tensor([0.229, 0.224, 0.225])
#
# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image file uploaded'}), 400
#
#         image_file = request.files['image']
#         if image_file.filename == '':
#             return jsonify({'error': 'No image file selected'}), 400
#
#         if not allowed_file(image_file.filename):
#             return jsonify({'error': 'Invalid file type. Allowed types: jpg, jpeg, png'}), 400
#
#         image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
#         image_file.save(image_path)
#
#         try:
#             # Open and preprocess the image
#             image = Image.open(image_path)
#             image = transform(image).unsqueeze(0)
#
#             # Normalize the image tensor
#             normalized_image = (image - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
#
#             # Perform prediction
#             with torch.no_grad():
#                 outputs = model(normalized_image)
#                 probabilities = torch.softmax(outputs, dim=1)
#                 class_idx = torch.argmax(probabilities, dim=1).item()
#                 prediction = classes[class_idx]
#                 probability = probabilities[0][class_idx].item() * 100
#
#             return jsonify({'prediction': prediction, 'probability': probability})
#         except Exception as e:
#             return jsonify({'error': str(e)}), 500
#
#     return jsonify({'error': 'Invalid request'})
#
# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify, render_template
# from torchvision import models, transforms
# from PIL import Image
# import torch
# import os
#
# app = Flask(__name__)
# UPLOAD_FOLDER = "images"
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}
#
# # Load the pre-trained model
# model = models.resnet50(pretrained=True)
# model.eval()
#
# # Define the transformations for the input data
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
# ])
#
# # Define class names (update as needed)
# classes = ['bacteria_Pneumonia', 'NORMAL','virus_Pneumonia']  # Assuming these are your classes
#
# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
#
# @app.route('/')
# def index():
#     return render_template('index.html')  # Assuming you have an index.html template
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image file uploaded'}), 400
#
#         image_file = request.files['image']
#         if image_file.filename == '':
#             return jsonify({'error': 'No image file selected'}), 400
#
#         if not allowed_file(image_file.filename):
#             return jsonify({'error': 'Invalid file type. Allowed types: jpg, jpeg, png'}), 400
#
#         image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
#         image_file.save(image_path)
#
#         # Open and preprocess the image
#         try:
#             image = Image.open(image_path)
#             image = transform(image).unsqueeze(0)  # Add batch dimension
#         except Exception as e:
#             return jsonify({'error': f'Error processing image: {str(e)}'}), 400
#
#         # Normalize the image tensor
#         normalized_image = (image - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
#
#         # Perform prediction
#         with torch.no_grad():
#             outputs = model(normalized_image)
#             probabilities = torch.softmax(outputs, dim=1)
#
#             # Ensure predicted class and probability are defined before formatting
#             if probabilities.shape[1] >= 2:  # Check for valid number of classes
#                 predicted_class_index = torch.argmax(probabilities, dim=1).item()
#
#                 # Check for valid index before accessing the list
#                 if 0 <= predicted_class_index < len(classes):
#                     predicted_class = classes[predicted_class_index]
#                     pneumonia_probability = probabilities[0][1].item() * 100
#                     formatted_probability = f"{pneumonia_probability:.2f}"  # Define here
#                 else:
#                     predicted_class = 'Unknown'
#                     formatted_probability = "N/A"  # Default value for missing probability
#             else:
#                 return jsonify({'error': 'Model output has unexpected number of classes'}), 400
#
#             return jsonify({'prediction': predicted_class, 'pneumonia_probability': formatted_probability})
#
#     return jsonify({'error': 'Invalid request'})
#
# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
from torchvision import models, transforms
from PIL import Image
import torch
import os

app = Flask(__name__)
UPLOAD_FOLDER = "images"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}

# Load the trained model
model = torch.load('pneumonia_model.pth')
model.eval()

# Define the transformations for the input data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class_names = ['Bacteria', 'Normal', 'Virus']  # Update class names


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        if not allowed_file(image_file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: jpg, jpeg, png'}), 400

        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
        image_file.save(image_path)

        # Open and preprocess the image
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        image = transform(image).unsqueeze(0)

        # Move the input tensor to the GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        image = image.to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            prediction = class_names[predicted[0].item()]
            confidence = confidence[0].item() * 100  # Convert to percentage

            print(f"Prediction: {prediction}, Confidence: {confidence:.2f}%")  # Debugging print statement

        return jsonify({'prediction': prediction, 'confidence': f'{confidence:.2f}%'})

    return jsonify({'error': 'Invalid request'})


if __name__ == '__main__':
    app.run(debug=True)
