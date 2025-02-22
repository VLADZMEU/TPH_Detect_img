from flask import Flask, request, render_template, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io
import json


app = Flask(__name__)

# Загрузка предобученной модели ResNet
model = models.resnet50(pretrained=True)
model.eval()


# Преобразование изображения для модели
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat.item()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        img_bytes = file.read()
        try:
            class_id = get_prediction(image_bytes=img_bytes)
            return jsonify({'class_id': class_id})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)