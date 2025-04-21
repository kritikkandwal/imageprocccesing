from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import plotly
from io import BytesIO
import plotly.graph_objs as go
import json
import joblib

app = Flask(__name__)

# Configure settings
DATASET_PATH = 'dataset/train'
MODEL_PATH = 'svm_model.joblib'
LABEL_ENCODER_PATH = 'label_encoder.joblib'
IMG_SIZE = (224, 224)

# Create feature extractor
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=x)

# Initialize SVM and Label Encoder
model = None
le = LabelEncoder()

def extract_features(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = feature_extractor.predict(x)
    return features.flatten()

def load_dataset():
    features = []
    labels = []
    
    for class_name in os.listdir(DATASET_PATH):
        class_dir = os.path.join(DATASET_PATH, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                feature = extract_features(img_path)
                features.append(feature)
                labels.append(class_name)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    return np.array(features), np.array(labels)

def train_model(kernel='linear', C=1.0):
    global model, le
    
    print("Loading dataset...")
    X, y = load_dataset()
    
    if len(X) == 0:
        raise ValueError("No training data found!")
    
    print("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("Training SVM...")
    model = svm.SVC(kernel=kernel, C=C, probability=True)
    model.fit(X, y_encoded)
    
    # Save model and encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print("Training complete!")

# Load pre-trained model if exists
if model is None or le is None:
    print("Initializing first-time training...")
    try:
        train_model()
        print("Initial training completed successfully!")
    except Exception as e:
        print(f"Initial training failed: {str(e)}")
        raise e


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    if model is None or le is None:
        return jsonify({'error': 'Model not trained!'})

    file = request.files['image']
    
    # Convert FileStorage to BytesIO
    img_bytes = file.read()
    img = image.load_img(BytesIO(img_bytes), target_size=IMG_SIZE)
    
    x = image.img_to_array(img)
    x = preprocess_input(np.expand_dims(x, axis=0))
    
    features = feature_extractor.predict(x)
    probs = model.predict_proba(features)[0]
    class_idx = model.predict(features)[0]
    class_name = le.inverse_transform([class_idx])[0]
    
    # Define top3 predictions
    top3_indices = probs.argsort()[-3:][::-1]
    top3 = [
        (le.inverse_transform([i])[0], float(probs[i]))
        for i in top3_indices
    ]
    
    return jsonify({
        'result': class_name,
        'confidence': float(probs[class_idx]),
        'top3': top3
    })

@app.route('/update_model', methods=['POST'])
def update_model():
    try:
        kernel = request.json.get('kernel', 'linear')
        C = float(request.json.get('C', 1.0))
        
        train_model(kernel=kernel, C=C)
        return jsonify({'status': 'success', 'message': 'Model retrained successfully!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)