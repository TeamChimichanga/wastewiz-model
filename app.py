from roboflow import Roboflow
from flask import Flask, request, jsonify

api_key = "8LSrlY9iHWxNrSMRud5u"
project_version = "rudo_v3"
version_num = 2

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predictResult():

    # image argument not passed
    if 'image' not in request.files:
        return 'No image part in the request', 400
    
    image = request.files['image']

    temp_image_path = 'temp_image.jpg'
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

    prediction = model.predict("temp_image.jpg", confidence=40, overlap=30)
    prediction.save("prediction.jpg")
    data = prediction.json()

    classes = set()
    for prediction in data.get("predictions", []):
        class_name = prediction.get("class")
        if class_name:
            classes.add(class_name)

    unique_classes = list(classes)

    result_image = open('prediction.jpg', 'rb')
    result = [result_image, unique_classes]

    return result, 200
