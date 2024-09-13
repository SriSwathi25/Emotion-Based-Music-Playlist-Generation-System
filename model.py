import random
import numpy as np
import pandas as pd
from keras import models, layers
from keras.utils.np_utils import to_categorical
from keras.models import load_model

def adjust_disgust(y_train, emotion_classes):
    """
    Consolidate 'Disgust' into 'Angry' category and update class indices.
    """
    print('Reassigning "Disgust" to "Angry"')
    y_train.loc[y_train == 1] = 0  # 'Disgust' is re-labeled as 'Angry'
    emotion_classes.remove('Disgust')
    emo_count = {}
    for idx, label in enumerate(emotion_classes):
        y_train.loc[y_train == emotion_dict[label]] = idx
        count = np.sum(y_train == idx)
        emo_count[label] = (idx, count)
    return y_train.values, emo_count

def load_emotion_data(sample_rate=0.3, data_usage='Training', selected_classes=['Angry', 'Happy'], data_path='./data/fer2013.csv'):
    """
    Load and preprocess data from a CSV file.
    """
    data_frame = pd.read_csv(data_path)
    data_frame = data_frame[data_frame.Usage == data_usage]
    selected_classes.append('Disgust')
    compiled_data = []
    for emotion in selected_classes:
        df_subset = data_frame[data_frame['emotion'] == emotion_dict[emotion]]
        compiled_data.append(df_subset)
    merged_data = pd.concat(compiled_data, ignore_index=True)
    sample_indices = random.sample(list(merged_data.index), int(len(merged_data) * sample_rate))
    sampled_data = merged_data.loc[sample_indices]
    
    pixel_data = [np.array([int(pixel) for pixel in instance.split()]) for instance in sampled_data["pixels"]]
    X = np.array(pixel_data).reshape(-1, 48, 48, 1).astype('float32') / 255
    y_train, class_distribution = adjust_disgust(sampled_data.emotion, selected_classes)
    y_train = to_categorical(y_train)
    return X, y_train

emotion_dict = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Loading datasets
X_test, y_test = load_emotion_data(1.0, 'PrivateTest', emotion_labels)
X_train, y_train = load_emotion_data(1.0, 'Training', emotion_labels)
X_val, y_val = load_emotion_data(1.0, 'PublicTest', emotion_labels)

def store_data(X, y, file_suffix=''):
    """
    Save arrays to files for future use.
    """
    np.save('X_test' + file_suffix, X)
    np.save('y_test' + file_suffix, y)

store_data(X_test, y_test, "_privatetest6_100pct")
X = np.load('X_test_privatetest6_100pct.npy')
y = np.load('y_test_privatetest6_100pct.npy')
print('Analysis of the private test set:')
emotion_count = np.bincount([np.argmax(label) for label in y])
print(dict(zip(emotion_labels, emotion_count)))

# Model Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(6, activation='softmax')
])

# Compiling and training the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=32, batch_size=128, validation_data=(X_val, y_val), shuffle=True, verbose=1)

model.save("model.h5")  # Save the trained model
