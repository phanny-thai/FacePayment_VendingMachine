# Deployment Guide: From Training to Production

This guide walks you through deploying your trained models from Jupyter notebook to the vending machine web application.

## 📋 Overview

Your training notebook (`Deeplearning_Project_Fix.ipynb`) generates:
- ✅ Model A: `modelA_classifier.keras`
- ✅ Model B: `modelB_siamese.keras`
- ✅ Encoder: `encoder.keras`
- ✅ Label classes: `label_classes.npy`

This guide shows how to use them in production.

## 🎯 Step-by-Step Deployment

### Phase 1: Training Models (Google Colab)

**You've already done this!** Your notebook trains both models. Here's what happens:

```python
# At the end of your notebook:
SAVE_DIR = "/content/drive/MyDrive/FacePaymentModels"

# Saves:
modelA.save(os.path.join(SAVE_DIR, "modelA_classifier.keras"))
modelB.save(os.path.join(SAVE_DIR, "modelB_siamese.keras"))
encoder.save(os.path.join(SAVE_DIR, "encoder.keras"))
np.save(os.path.join(SAVE_DIR, "label_classes.npy"), le.classes_)
```

### Phase 2: Download Model Files

**From Google Drive:**

1. Navigate to: `MyDrive/FacePaymentModels/`
2. Download these files:
   - `modelA_classifier.keras` (~10MB)
   - `modelB_siamese.keras` (~15MB)
   - `encoder.keras` (~10MB)
   - `label_classes.npy` (~1KB)

**Or using gdown (CLI):**

```bash
# Install gdown
pip install gdown

# Download from shared Google Drive link
gdown YOUR_SHARED_LINK_HERE
```

### Phase 3: Setup Project Directory

```bash
# Create project folder
mkdir face_vending_machine
cd face_vending_machine

# Place downloaded files
mv ~/Downloads/modelA_classifier.keras .
mv ~/Downloads/modelB_siamese.keras .
mv ~/Downloads/encoder.keras .
mv ~/Downloads/label_classes.npy .

# Download application files from Claude outputs
# - app_triple_model.py
# - model.py
# - index_triple.html
# - README_TRIPLE_MODEL.md
```

**Final structure:**
```
face_vending_machine/
├── app_triple_model.py
├── model.py
├── index_triple.html
├── modelA_classifier.keras       ← Your trained model
├── modelB_siamese.keras          ← Your trained model
├── encoder.keras                 ← Your trained encoder
├── label_classes.npy             ← Your class names
└── requirements_triple.txt
```

### Phase 4: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_triple.txt

# Or install manually
pip install flask tensorflow torch torchvision pillow numpy
```

### Phase 5: Test Models Load Correctly

```bash
# Quick test
python -c "
import tensorflow as tf
import numpy as np

# Test Model A
print('Loading Model A...')
modelA = tf.keras.models.load_model('modelA_classifier.keras')
print(f'✅ Model A loaded: {modelA.input_shape}')

# Test Model B
print('Loading Model B...')
modelB = tf.keras.models.load_model('modelB_siamese.keras')
print(f'✅ Model B loaded: {modelB.input_shape}')

# Test encoder
print('Loading Encoder...')
encoder = tf.keras.models.load_model('encoder.keras')
print(f'✅ Encoder loaded: {encoder.output_shape}')

# Test classes
print('Loading Classes...')
classes = np.load('label_classes.npy', allow_pickle=True)
print(f'✅ Classes loaded: {len(classes)} classes')
print(f'   Sample classes: {list(classes[:5])}')
"
```

Expected output:
```
Loading Model A...
✅ Model A loaded: (None, 160, 160, 3)
Loading Model B...
✅ Model B loaded: [(None, 160, 160, 3), (None, 160, 160, 3)]
Loading Encoder...
✅ Encoder loaded: (None, 128)
Loading Classes...
✅ Classes loaded: 20 classes
   Sample classes: ['Person_01', 'Person_02', 'Person_03', ...]
```

### Phase 6: Start Application

```bash
# Start the server
python app_triple_model.py
```

Expected console output:
```
======================================================================
🤖 Face Payment Vending Machine (Triple Model Support)
======================================================================
Active Model: MODELA
PyTorch Siamese:      ❌ Not loaded
Model A (Classifier): ✅ Loaded
Model B (Siamese):    ✅ Loaded
Encoder (for Model B): ✅ Loaded
======================================================================

 * Running on http://0.0.0.0:5000
```

### Phase 7: Test in Browser

1. Open: `http://localhost:5000`
2. Click "Start Camera" (allow permissions)
3. Register yourself:
   - Capture face
   - Enter your name (must match one of the training classes!)
   - Set balance: $10
   - Click "Register Face"

4. Test verification:
   - Go to "Purchase" tab
   - Capture face
   - Leave User ID empty (Model A auto-identifies)
   - Click "Verify Face"
   - Should identify you correctly!

## 🔧 Configuration Tips

### Switching Between Models

**Use Model A (auto-identification):**
```python
# In app_triple_model.py
MODEL_TYPE = 'modelA'  # Fast, convenient
```

**Use Model B (secure verification):**
```python
# In app_triple_model.py
MODEL_TYPE = 'modelB'  # High accuracy, requires User ID
```

**Auto-select best available:**
```python
# In app_triple_model.py
MODEL_TYPE = 'auto'  # System chooses automatically
```

### Adjusting Thresholds

**Model A - Confidence threshold:**
```python
CONFIDENCE_THRESHOLD = 0.5  # Range: 0.0 - 1.0
# 0.5 = 50% confidence minimum
# 0.7 = 70% confidence (stricter)
# 0.3 = 30% confidence (more lenient)
```

**Model B - Similarity threshold:**
```python
THRESHOLD_SIAMESE_KERAS = 0.5  # Range: 0.0 - 1.0
# 0.5 = 50% similarity minimum
# 0.7 = 70% similarity (stricter)
# 0.3 = 30% similarity (more lenient)
```

## 🎓 Understanding Class Mapping

Your notebook uses **folder names as class labels**. For example:

```
Dataset/
├── Person_01/  → Class 0
├── Person_02/  → Class 1
├── Person_03/  → Class 2
└── ...
```

**Important:**
- Users must register with names matching training classes
- Or update `label_classes.npy` to include new users

### Adding New Users After Training

**Option 1: Fine-tune Model A (Recommended)**

```python
# In your notebook, add new user images
# Then fine-tune the last layer:

modelA.layers[-1].trainable = True
modelA.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),  # Lower learning rate
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train for a few more epochs with new + old data
history = modelA.fit(train_ds, epochs=5)

# Save updated model
modelA.save("modelA_classifier_updated.keras")
```

**Option 2: Use Model B (No retraining needed)**

Model B generates embeddings - no retraining required for new users!

```bash
# Switch to Model B in config
MODEL_TYPE = 'modelB'
```

## 🚀 Production Deployment

### Option 1: Local Network (Vending Machine)

```bash
# Find your local IP
# On Linux/Mac:
ifconfig | grep "inet "
# On Windows:
ipconfig

# Example output: 192.168.1.100

# Start server on all interfaces
python app_triple_model.py
# Now accessible at: http://192.168.1.100:5000
```

### Option 2: Cloud Deployment (AWS/GCP/Azure)

**Using Gunicorn (recommended for production):**

```bash
# Install gunicorn
pip install gunicorn

# Start with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 app_triple_model:app
```

**Docker deployment:**

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements_triple.txt .
RUN pip install --no-cache-dir -r requirements_triple.txt

COPY . .
EXPOSE 5000

CMD ["python", "app_triple_model.py"]
```

```bash
# Build and run
docker build -t face-vending .
docker run -p 5000:5000 face-vending
```

### Option 3: Raspberry Pi / Edge Device

```bash
# On Raspberry Pi 4 (4GB RAM minimum)

# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-opencv

# Install Python packages
pip3 install flask tensorflow-cpu pillow numpy

# Copy project files
# Start server
python3 app_triple_model.py
```

## 🔐 Security Hardening (Production)

### 1. Add Authentication

```python
# In app_triple_model.py, add:
from functools import wraps
from flask import request, abort

API_KEY = "your-secret-key-here"

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key')
        if key != API_KEY:
            abort(401)
        return f(*args, **kwargs)
    return decorated

@app.route('/register', methods=['POST'])
@require_api_key  # Add this
def register():
    # ... existing code
```

### 2. Enable HTTPS

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365

# Run with HTTPS
python app_triple_model.py --cert cert.pem --key key.pem
```

Or modify app:
```python
if __name__ == '__main__':
    app.run(
        debug=False,  # Disable debug in production
        host='0.0.0.0',
        port=5000,
        ssl_context=('cert.pem', 'key.pem')
    )
```

### 3. Rate Limiting

```bash
pip install flask-limiter
```

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/verify', methods=['POST'])
@limiter.limit("10 per minute")
def verify():
    # ... existing code
```

## 📊 Monitoring & Logging

```python
# Add to app_triple_model.py
import logging
from datetime import datetime

logging.basicConfig(
    filename='vending_machine.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Log each transaction
@app.route('/purchase', methods=['POST'])
def purchase():
    # ... verification code ...
    
    if purchase_successful:
        logging.info(f"Purchase: {user_id} bought {item_name} for ${price}")
    
    # ... rest of code
```

## 🐛 Common Issues & Solutions

### Issue 1: "No module named 'tensorflow'"

**Solution:**
```bash
pip uninstall tensorflow
pip install tensorflow==2.15.0
```

### Issue 2: Model predictions are wrong

**Solution:**
Check if registered users match training classes:
```python
# Verify classes
import numpy as np
classes = np.load('label_classes.npy', allow_pickle=True)
print("Trained on:", list(classes))

# Users must register with these exact names!
```

### Issue 3: "Encoder not found" for Model B

**Solution:**
```bash
# Verify files exist
ls -lh *.keras *.npy

# Should see:
# modelA_classifier.keras
# modelB_siamese.keras
# encoder.keras
# label_classes.npy
```

### Issue 4: Camera not working

**Solution:**
- Use HTTPS or localhost
- Check browser permissions
- Try different browser (Chrome works best)

## 📈 Performance Optimization

### 1. Model Quantization (Faster inference)

```python
# In your training notebook:
converter = tf.lite.TFLiteConverter.from_keras_model(modelA)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('modelA_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 2. Batch Processing

```python
# Process multiple faces at once
def verify_batch(images, user_ids):
    # Batch predictions are faster
    results = []
    for img, uid in zip(images, user_ids):
        result = verify_modelA(img, uid)
        results.append(result)
    return results
```

### 3. Caching Embeddings (Model B)

Already implemented! Embeddings are cached in `user_embeddings.json`.

## ✅ Deployment Checklist

- [ ] Models trained and saved
- [ ] Files downloaded from Google Drive
- [ ] Project directory set up correctly
- [ ] Dependencies installed
- [ ] Models load without errors
- [ ] Server starts successfully
- [ ] Camera access works
- [ ] User registration works
- [ ] Face verification works
- [ ] Purchase transaction works
- [ ] Tested with all trained users
- [ ] Thresholds adjusted for accuracy
- [ ] Logs configured
- [ ] Security measures in place (if production)
- [ ] Backup of model files created

## 🎉 You're Done!

Your face payment vending machine is now ready for deployment!

**Next Steps:**
1. Test thoroughly with real users
2. Collect feedback on accuracy
3. Fine-tune thresholds as needed
4. Consider adding liveness detection
5. Monitor performance and logs

---

**Need help?** Check the README or troubleshooting guide!
