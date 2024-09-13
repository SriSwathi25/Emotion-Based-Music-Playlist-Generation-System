from keras.models import load_model
from sklearn.metrics import accuracy_score
import numpy as np

model = load_model('model.h5')
X_fname = 'X_test_privatetest6_100pct.npy'
y_fname = 'y_test_privatetest6_100pct.npy'
X = np.load(X_fname)
y = np.load(y_fname)
y_labels = [np.argmax(lst) for lst in y]
counts = np.bincount(y_labels)
labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
print (zip(labels, counts))

test_pred = np.argmax(model.predict(X), axis=1)
print("CNN Model Accuracy on test set: {:.4f}".format(accuracy_score(y_labels, test_pred)))
