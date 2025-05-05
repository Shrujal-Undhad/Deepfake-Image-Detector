from flask import Flask, request, jsonify, render_template
import torch
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import joblib
import os
import json
from datetime import datetime
import threading

# Initialize app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATHS = {
    'cnn': 'models/efficientnet_model.pt',
    'xgboost': 'models/xgboost_model.json',
    'yolo': 'models/yolov8n.pt'
}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Metrics config
METRICS_FILE = 'metrics/metrics.json'
METRICS_LOCK = threading.Lock()
os.makedirs('metrics', exist_ok=True)

# Global device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
def load_models():
    models = {}

    import glob
    model_files = glob.glob('models/*')
    print(f"Found model files: {model_files}")  # Debug print

    for model_file in model_files:
        try:
            print(f"Processing model file: {model_file}")  # Debug print

            if model_file.endswith('.pt') and 'yolo' in model_file:
                print("Loading YOLO model...")  # Debug print
                models['yolo'] = YOLO(model_file)
                print("YOLO model loaded successfully.")  # Debug print

            elif model_file.endswith('.pt'):
                print("Loading EfficientNet model...")  # Debug print
                from efficientnet_pytorch import EfficientNet
                model = EfficientNet.from_name('efficientnet-b0')
                model._fc = torch.nn.Sequential(torch.nn.Linear(model._fc.in_features, 1))

                print(f"Loading model weights from {model_file}...")  # Debug print
                model.load_state_dict(torch.load(model_file, map_location=DEVICE))
                model.to(DEVICE)
                model.eval()
                models['cnn'] = model
                print("EfficientNet model loaded successfully.")  # Debug print

            elif model_file.endswith('.json'):
                print("Loading XGBoost model...")  # Debug print
                import xgboost as xgb
                booster = xgb.Booster()
                booster.load_model(model_file)
                models['xgboost'] = booster
                print("XGBoost model loaded successfully.")  # Debug print

        except Exception as e:
            print(f"Error loading model from {model_file}: {str(e)}")
            continue

    return models

# Load models globally
models = load_models()

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).float().unsqueeze(0)

def extract_features(imgs):
    feats = []
    for img in imgs:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        feats.append([
            np.mean(gray),
            np.std(gray),
            cv2.Laplacian(gray, cv2.CV_64F).var(),
            cv2.Sobel(gray, cv2.CV_64F, 1, 0).var(),
            cv2.Sobel(gray, cv2.CV_64F, 0, 1).var()
        ])
    return np.array(feats)

def update_metrics(is_deepfake, confidence):
    with METRICS_LOCK:
        metrics = {
            'total_images': 0,
            'deepfake_images': 0,
            'authentic_images': 0,
            'average_confidence': 0.0
        }

        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, 'r') as f:
                metrics = json.load(f)

        metrics['total_images'] += 1
        if is_deepfake:
            metrics['deepfake_images'] += 1
        else:
            metrics['authentic_images'] += 1

        total_conf = metrics['average_confidence'] * (metrics['total_images'] - 1)
        total_conf += confidence
        metrics['average_confidence'] = total_conf / metrics['total_images']

        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=4)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            image = cv2.imread(filepath)
            image_tensor = preprocess_image(image).to(DEVICE)  # <<< FIXED: Move to model device

            # CNN prediction
            cnn_prob, cnn_pred = 0.0, False
            if 'cnn' in models:
                with torch.no_grad():
                    output = models['cnn'](image_tensor)
                    cnn_prob = torch.sigmoid(output).item()
                    cnn_pred = cnn_prob > 0.6

            # XGBoost prediction
            xgb_prob, xgb_pred = 0.0, False
            if 'xgboost' in models:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                features = extract_features([img_rgb])
                import xgboost as xgb
                dmatrix = xgb.DMatrix(features)
                preds = models['xgboost'].predict(dmatrix)
                xgb_prob = preds[0]
                xgb_pred = xgb_prob > 0.6

            # YOLO detection
            yolo_prob = 0.0
            result_path = None
            if 'yolo' in models:
                yolo_results = models['yolo'](filepath)
                if len(yolo_results[0].boxes) > 0:
                    yolo_prob = 0.5
                    annotated_img = yolo_results[0].plot()
                    result_path = os.path.join(UPLOAD_FOLDER, f'result_{filename}')
                    cv2.imwrite(result_path, annotated_img)

            is_deepfake = bool(cnn_pred or xgb_pred)
            confidence = float(max(cnn_prob, xgb_prob))

            update_metrics(is_deepfake, confidence)

            return jsonify({
                'is_deepfake': is_deepfake,
                'confidence': confidence,
                'result_path': f'/static/uploads/result_{filename}' if result_path else None,
                'message': 'Deepfake detected' if is_deepfake else 'Authentic image'
            })

        except Exception as e:
            print(f"Error processing(/detect) image: {str(e)}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/dashboard')
def dashboard():
    try:
        with open(METRICS_FILE) as f:
            metrics = json.load(f)
        return render_template('dashboard.html', metrics=metrics)
    except FileNotFoundError:
        return render_template('dashboard.html', metrics=None)

@app.route('/api/metrics')
def get_metrics():
    try:
        with open(METRICS_FILE) as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({'error': 'Metrics not available'}), 404

@app.route('/api/gpu-status')
def gpu_status():
    status = {
        'gpu_available': torch.cuda.is_available(),
        'device': str(DEVICE)
    }
    if torch.cuda.is_available():
        status.update({
            'memory_used': f"{torch.cuda.memory_allocated()/1024**3:.1f}GB",
            'memory_total': f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB"
        })
    return jsonify(status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
