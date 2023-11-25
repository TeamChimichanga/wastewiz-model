from roboflow import Roboflow
from flask import Flask, request, jsonify
import base64
from dotenv import load_dotenv
import os

project_version = "rudo_v3"
version_num = 2
prediction_img_name = "prediction.jpg"
temp_img_name = "temp.jpg"

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predictResult():

    load_dotenv()
    api_key = os.getenv("API_KEY")
    # image argument not passed
    if 'image' not in request.files:
        return 'No image part in the request', 400
    
    image = request.files['image']

    temp_image_path = temp_img_name
    image.save(temp_image_path)

    # invalid name
    if image.filename == '':
        return 'No selected file', 400
    
    # invalid extension
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' not in image.filename or image.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return 'Invalid file format', 400

    rf = Roboflow(api_key)
    project = rf.workspace().project(project_version)
    model = project.version(version_num).model

    prediction = model.predict(temp_image_path, confidence=40, overlap=30)
    prediction.save(prediction_img_name)
    data = prediction.json()

    classes = set()
    for prediction in data.get("predictions", []):
        class_name = prediction.get("class")
        if class_name:
            classes.add(class_name)

    unique_classes = list(classes)

    # Open the result image and convert to base64
    with open(prediction_img_name, "rb") as img_file:
        result_image = base64.b64encode(img_file.read()).decode('utf-8')

    # Return the result image as base64 string and unique classes detected
    result = {'result_image_base64': result_image, 'unique_classes': unique_classes}

    return result, 200
