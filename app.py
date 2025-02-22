from flask import Flask, request, render_template, jsonify
import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import io
import json

app = Flask(__name__)

# Загрузка предобученной модели ResNet
model = models.resnet50(pretrained=True)
model.eval()

# Загрузка меток классов ImageNet
with open('imagenet-simple-labels.json', 'r', encoding='utf-8') as f:
    labels = json.load(f)

# Преобразование class_id в название класса
def class_id_to_label(class_id):
    return labels[class_id]

# Преобразование изображения для модели
def transform_image(image_bytes):
    try:
        # Открываем изображение и конвертируем в RGB, если оно RGBA
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        my_transforms = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )])
        return my_transforms(image).unsqueeze(0)
    except UnidentifiedImageError:
        raise ValueError("Загруженный файл не является изображением или формат не поддерживается.")
    except Exception as e:
        raise ValueError(f"Ошибка при обработке изображения: {str(e)}")

# Получение предсказания
def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        outputs = model.forward(tensor)
        _, y_hat = outputs.max(1)
        return y_hat.item()
    except Exception as e:
        raise ValueError(f"Ошибка при классификации изображения: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Проверка наличия файла
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не найден. Пожалуйста, загрузите изображение.'}), 400
        file = request.files['file']
        if not file:
            return jsonify({'error': 'Файл не найден. Пожалуйста, загрузите изображение.'}), 400

        # Проверка формата файла
        if file.filename == '':
            return jsonify({'error': 'Имя файла пустое. Пожалуйста, выберите файл.'}), 400

        # Чтение файла
        img_bytes = file.read()
        if not img_bytes:
            return jsonify({'error': 'Файл пустой. Пожалуйста, загрузите корректное изображение.'}), 400

        try:
            # Получение предсказания
            class_id = get_prediction(image_bytes=img_bytes)
            class_name = class_id_to_label(class_id)
            return jsonify({'class_name': class_name})
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': f"Неизвестная ошибка: {str(e)}"}), 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)