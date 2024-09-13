from keras.models import load_model
import base64
import io
import os
import cv2
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for, redirect, session

# Initializing the Flask application
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1  # Set default send file max age

# Load the model and Haar cascade for face detection
print("+"*50, "Model loading in progress")
model = load_model('model.h5')
haarcascade = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(haarcascade)

# Define emotion labels
label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']

@app.route('/')
def index():
    # Render the main page
    return render_template('index.html')

@app.route('/music-player')
def music_player():
    # Music player page
    prediction = request.args.get('prediction')
    song = request.args.get('song')
    return render_template('music_player.html', prediction=prediction, song=song)

@app.route('/songs_list/<prediction>')
def songs_list(prediction):
    # List songs based on emotion
    f_path = 'static/' + prediction
    folder_contents = os.listdir(f_path)
    return render_template('songs_list.html', folder_contents=folder_contents, prediction=prediction)

@app.route('/process-image', methods=['POST'])
def process_image():
    # Process the uploaded image and predict emotion
    data = json.loads(request.data)
    dataURL = data.get('image_data')
    image_data = dataURL.split(',')[1]
    decoded_data = base64.b64decode(image_data)
    np_data = np.frombuffer(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.4, 1)
    for x, y, w, h in faces:
        roi = gray[y:y+h, x:x+w]
    try:
        roi = cv2.resize(roi, (48, 48))
        roi = roi/255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        prediction = model.predict(roi)
        prediction = np.argmax(prediction)
        prediction = label_map[prediction]
        print("Emotion Detected:", prediction)
        return redirect(url_for('songs_list', prediction=prediction))
    except Exception as e:
        print("Error in processing image:", e)
        return redirect(url_for('error'))

@app.route('/error-page')
def error():
    # Display error page
    return render_template('error_page.html')

# Run the app
app.run(host='0.0.0.0', port=5000, debug=True)
