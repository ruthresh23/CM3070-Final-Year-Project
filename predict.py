import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("model/breast_cancer_model.h5")

img_size = 224

# Class names (same order as dataset folders)
classes = ["Benign", "Malignant", "Normal"]

def predict_image(path):

    img = cv2.imread(path)
    img = cv2.resize(img, (img_size, img_size))

    img = img / 255.0
    img = np.reshape(img, (1, img_size, img_size, 3))

    prediction = model.predict(img)[0]

    print("Prediction probabilities:", prediction)

    class_index = np.argmax(prediction)

    return classes[class_index]