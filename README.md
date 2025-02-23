# TSI_Detect_img
# Emotion Classification Model

This project is an emotion classification model based on text, using LSTM and pretrained embeddings. The model is trained on textual data and predicts the emotions expressed in those texts.

## Installation

To run the project, you will need the following libraries:
```bash
pip install numpy pandas keras tensorflow nltk matplotlib
```


You also need to download some NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```


## Data

The training, validation, and test data are loaded from the files `train.txt`, `val.txt`, and `test.txt`, respectively. The files should contain text and emotion labels separated by a semicolon.

### Data Format

The data files should be in the following format:
```
Text;Emotion
```


## Text Processing

The text is processed using the following steps:
1. Removal of URLs.
2. Removal of non-alphabetic characters.
3. Conversion of text to lowercase.
4. Removal of stop words.
5. Lemmatization of words.

## Model Training

The model is a sequential network using LSTM. It is compiled with the Adam optimizer and the loss function `sparse_categorical_crossentropy`. The model is trained for 10 epochs with a callback for reducing the learning rate.

## Model Saving

After training, the model and tokenizer are saved in HDF5 and pickle formats, respectively:
```python
model.save("emotion_model.h5")
with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)
with open("encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)
```


## Prediction Example

To make a prediction on new text, you can use the following code:
```python
input_text = "I feel hurt because he cheated on me with his best friend"
input_sequence = tokenizer.texts_to_sequences([input_text])
padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
prediction = model.predict(padded_input_sequence)
predicted_label = encoder.inverse_transform([np.argmax(prediction[0])])
print(f"Predicted emotion: {predicted_label[0]}")
```


## License

This project is licensed under the MIT License.
Feel free to modify or expand upon this README as needed for your project!

Question :
```bash
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from sklearn.preprocessing import LabelEncoder 
import numpy as np 
import re 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
import pickle 
import asyncio 
from telegram import Update 
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes 
import nest_asyncio 
nest_asyncio.apply() 
nltk.download('stopwords') 
nltk.download('wordnet') 
nltk.download('omw-1.4') 
# Функция для очистки текста 
def normalize_text(text): 
    text = re.sub(r'https?://\S+|www\.\S+','', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = text.lower() 
    stop_words = set(stopwords.words('english')) 
    text = " ".join(word for word in text.split() if word not in stop_words) 
    lemmatizer = WordNetLemmatizer() 
    text = " ".join(lemmatizer.lemmatize(word) for word in text.split()) 
    return text 
# Загрузка модели 
def load_model(model_path): 
    model = tf.keras.models.load_model(model_path) 
    return model 
# Подготовка текста 
def prepare_text(text, tokenizer, max_length): 
    text = normalize_text(text) 
    sequence = tokenizer.texts_to_sequences([text]) 
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post') 
    return padded_sequence 
# Предсказание эмоции с вероятностями для всех классов 
def predict_emotion_with_probability(model, text, tokenizer, max_length, encoder): 
    padded_sequence = prepare_text(text, tokenizer, max_length) 
    prediction = model.predict(padded_sequence) 
    probabilities = prediction[0]  # Вероятности для всех классов 
    predicted_index = np.argmax(probabilities)  # Индекс предсказанного класса 
    # Преобразуем индекс в текстовую метку 
    predicted_label = encoder.inverse_transform([predicted_index])[0]  # Предсказанная эмоция 
    return predicted_label, probabilities 
# Обработчик команды /start 
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None: 
    await update.message.reply_text('Привет! Я бот для анализа настроения текста. Отправь мне текст, и я скажу, какое у него настроение.') 
# Обработчик текстовых сообщений 
async def send_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None: 
    model_path = "emotion_model.h5" 
    model = load_model(model_path) 
    tokenizer = pickle.load(open("tokenizer.pkl", "rb")) 
    encoder = pickle.load(open("encoder.pkl", "rb")) 
    max_length = 40 
    text = update.message.text 
    # Получаем эмоцию и вероятности для всех классов 
    emotion, probabilities = predict_emotion_with_probability(model, text, tokenizer, max_length, encoder) 
    # Формируем ответ с дополнительной информацией 
    response = f"Настроение текста: **{emotion}**\n\n"  # Основная эмоция 
    # Вероятности для всех эмоций 
    response += "Вероятности для всех эмоций:\n" 
    for i, label in enumerate(encoder.classes_): 
        response += f"- {label}: {probabilities[i]:.2f}\n" 
    # Топ-3 наиболее вероятных эмоций 
    top_indices = np.argsort(probabilities)[-3:][::-1]  # Индексы топ-3 эмоций 
    response += "\nТоп-3 наиболее вероятных эмоций:\n" 
    for idx in top_indices: 
        response += f"- {encoder.classes_[idx]}: {probabilities[idx]:.2f}\n" 
    # Дополнительная информация 
    response += f"\nДлина текста: {len(text.split())} слов\n" 
    response += f"Очищенный текст: {normalize_text(text)}\n" 
    # Отправка ответа 
    await update.message.reply_text(response) 
# Основная функция 
``` bash
async def main() -> None: 
    application = ApplicationBuilder().token("8143120396:AAHqHWcdpkj7ApJ-BwXLlZuIA5Cwaxvw8t4").build() 
    application.add_handler(CommandHandler("start", start)) 
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, send_text)) 
    await application.run_polling() 
if __name__ == '__main__': 
    loop = asyncio.get_event_loop() 
    loop.run_until_complete(main()) Ещё для этого
```
# Telegram Bot for Emotion Analysis

This project is a Telegram bot that analyzes the sentiment of text messages using a trained emotion classification model. The bot predicts the emotion expressed in the text and provides probabilities for all possible emotions.

## Installation

To run the bot, you will need the following libraries:
```bash
pip install tensorflow numpy nltk python-telegram-bot
```

You also need to download some NLTK resources:
```bash
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```


## Data and Model

Make sure you have the following files in the same directory as your script:
- `emotion_model.h5`: The trained model for emotion classification.
- `tokenizer.pkl`: The tokenizer used for text preprocessing.
- `encoder.pkl`: The label encoder that maps emotion labels to integers.

## How to Use

1. Clone this repository or download the files.
2. Install the required libraries using the command provided above.
3. Run the script:
```bash
python your_script_name.py
```

4. Open Telegram and find your bot using its username.
5. Start a conversation with the bot by sending the `/start` command.

## Bot Functionality

- **/start**: Sends a welcome message and instructions on how to use the bot.
- **Text Analysis**: Send any text to the bot, and it will respond with:
  - The predicted emotion.
  - Probabilities for all emotions.
  - The top 3 most likely emotions.
  - The length of the text in words.
  - The cleaned text after normalization.

## Example Interaction
```
User: I feel hurt because he cheated on me with his best friend.
Bot: 
Настроение текста: sadness

Вероятности для всех эмоций:

happiness: 0.05
sadness: 0.85
anger: 0.10
Топ-3 наиболее вероятных эмоций:

sadness: 0.85
anger: 0.10
happiness: 0.05
Длина текста: 10 слов
Очищенный текст: feel hurt cheated best friend
