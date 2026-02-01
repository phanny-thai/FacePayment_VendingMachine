from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import base64
import json
import os
import numpy as np
from model import SiameseNetwork

# Try to import TensorFlow/Keras (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️  TensorFlow not available. Keras models disabled.")

app = Flask(__name__, static_folder='.')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# ============================================
# CONFIGURATION
# ============================================
# Model file paths
SIAMESE_PYTORCH_PATH = 'siamese_modelnew.pth'
MODEL_A_PATH = 'modelA_classifier.keras'  # MobileNetV2 Classifier
MODEL_B_PATH = 'modelB_siamese.keras'     # MobileNetV2 Siamese
ENCODER_PATH = 'encoder.keras'            # For Model B
LABEL_CLASSES_PATH = 'label_classes.npy'  # Class names for Model A

# Data paths
REGISTERED_USERS_PATH = 'registered_users.json'
USER_ID_MAPPING_PATH = 'user_id_mapping.json'
USER_EMBEDDINGS_PATH = 'user_embeddings.json'  # For Model B embeddings

# Thresholds
THRESHOLD_SIAMESE_PYTORCH = 1.0  # L2 distance threshold
THRESHOLD_SIAMESE_KERAS = 0.7    # Similarity score threshold (0-1)
CONFIDENCE_THRESHOLD = 0.7       # For Model A predictions

# Image sizes
IMG_SIZE_PYTORCH = (105, 105)
IMG_SIZE_KERAS = (160, 160)

# Model selection: 'auto', 'siamese_pytorch', 'modelA', 'modelB'
MODEL_TYPE = 'siamese_pytorch'

# ============================================
# INITIALIZE MODELS
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model storage
siamese_pytorch = None
modelA_classifier = None
modelB_siamese = None
encoder_model = None
label_classes = None
user_id_to_class = {}
class_to_user_id = {}
user_embeddings = {}

# Load PyTorch Siamese
if os.path.exists(SIAMESE_PYTORCH_PATH):
    try:
        siamese_pytorch = SiameseNetwork().to(device)
        checkpoint = torch.load(SIAMESE_PYTORCH_PATH, map_location=device)
        siamese_pytorch.load_state_dict(checkpoint['model_state_dict'])
        siamese_pytorch.eval()
        print("✅ PyTorch Siamese model loaded")
    except Exception as e:
        print(f"❌ Error loading PyTorch Siamese: {e}")

# Load Keras models
if KERAS_AVAILABLE:
    # Model A: MobileNetV2 Classifier
    if os.path.exists(MODEL_A_PATH):
        try:
            modelA_classifier = keras.models.load_model(MODEL_A_PATH)
            print("✅ Model A (MobileNetV2 Classifier) loaded")
            
            # Load label classes
            if os.path.exists(LABEL_CLASSES_PATH):
                label_classes = np.load(LABEL_CLASSES_PATH, allow_pickle=True)
                print(f"   Loaded {len(label_classes)} classes: {list(label_classes[:5])}")
            
            # Load user ID mapping
            if os.path.exists(USER_ID_MAPPING_PATH):
                with open(USER_ID_MAPPING_PATH, 'r') as f:
                    mapping = json.load(f)
                    user_id_to_class = mapping.get('user_id_to_class', {})
                    class_to_user_id = mapping.get('class_to_user_id', {})
        except Exception as e:
            print(f"❌ Error loading Model A: {e}")
    
    # Model B: MobileNetV2 Siamese
    if os.path.exists(MODEL_B_PATH):
        try:
            modelB_siamese = keras.models.load_model(MODEL_B_PATH)
            print("✅ Model B (MobileNetV2 Siamese) loaded")
            
            # Load encoder for embeddings
            if os.path.exists(ENCODER_PATH):
                encoder_model = keras.models.load_model(ENCODER_PATH)
                print("✅ Encoder model loaded")
            
            # Load pre-computed embeddings
            if os.path.exists(USER_EMBEDDINGS_PATH):
                with open(USER_EMBEDDINGS_PATH, 'r') as f:
                    user_embeddings = json.load(f)
                print(f"   Loaded embeddings for {len(user_embeddings)} users")
        except Exception as e:
            print(f"❌ Error loading Model B: {e}")

# Determine active model
if MODEL_TYPE == 'auto':
    if modelA_classifier is not None:
        MODEL_TYPE = 'modelA'
        print("🤖 Using Model A (MobileNetV2 Classifier)")
    elif modelB_siamese is not None:
        MODEL_TYPE = 'modelB'
        print("🤖 Using Model B (MobileNetV2 Siamese)")
    elif siamese_pytorch is not None:
        MODEL_TYPE = 'siamese_pytorch'
        print("🤖 Using PyTorch Siamese Network")
    else:
        print("❌ No model available!")

# ============================================
# IMAGE PROCESSING FUNCTIONS
# ============================================
transform_pytorch = transforms.Compose([
    transforms.Resize(IMG_SIZE_PYTORCH),
    transforms.ToTensor(),
])

def preprocess_keras(image):
    """Preprocess image for Keras models (MobileNetV2)"""
    image = image.convert('RGB')
    image = image.resize(IMG_SIZE_KERAS)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def process_image_pytorch(image_data):
    """Process base64 image for PyTorch"""
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    return transform_pytorch(image).unsqueeze(0).to(device)

def process_image_keras(image_data):
    """Process base64 image for Keras"""
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    return preprocess_keras(image)

# ============================================
# VERIFICATION FUNCTIONS
# ============================================
def verify_pytorch_siamese(captured_image, user_id):
    """Verify using PyTorch Siamese Network"""
    if user_id not in registered_users:
        return False, float('inf'), "User not registered"
    
    registered_image = registered_users[user_id]
    
    with torch.no_grad():
        img1_tensor = process_image_pytorch(captured_image)
        img2_tensor = process_image_pytorch(registered_image)
        
        out1, out2 = siamese_pytorch(img1_tensor, img2_tensor)
        distance = F.pairwise_distance(out1, out2).item()
        
        is_verified = distance < THRESHOLD_SIAMESE_PYTORCH
        confidence = max(0, 1 - (distance / THRESHOLD_SIAMESE_PYTORCH)) * 100
        
        return is_verified, distance, f"Confidence: {confidence:.1f}%"

def verify_modelA(captured_image, user_id=None):
    """Verify/identify using Model A (Classifier)"""
    img_array = process_image_keras(captured_image)
    
    # Get prediction
    predictions = modelA_classifier.predict(img_array, verbose=0)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100
    
    # Get user ID from predicted class
    if label_classes is not None and predicted_class < len(label_classes):
        predicted_user_id = str(label_classes[predicted_class])
    else:
        predicted_user_id = class_to_user_id.get(str(predicted_class), None)
    
    if user_id:
        # Verification mode
        is_verified = (predicted_user_id == user_id) and (confidence > CONFIDENCE_THRESHOLD * 100)
        return is_verified, confidence, f"Predicted: {predicted_user_id}, Confidence: {confidence:.1f}%"
    else:
        # Identification mode
        return True, confidence, predicted_user_id

def verify_modelB(captured_image, user_id):
    """Verify using Model B (Siamese with embeddings)"""
    if user_id not in user_embeddings:
        return False, 0.0, "User not registered or no embedding"
    
    # Get embedding of captured image
    img_array = process_image_keras(captured_image)
    captured_embedding = encoder_model.predict(img_array, verbose=0)
    
    # Get stored embedding
    stored_embedding = np.array(user_embeddings[user_id])
    stored_embedding = np.expand_dims(stored_embedding, axis=0)
    
    # Compute similarity using Model B
    similarity = modelB_siamese.predict([stored_embedding, captured_embedding], verbose=0)[0][0]
    
    is_verified = similarity > THRESHOLD_SIAMESE_KERAS
    confidence = similarity * 100
    
    return is_verified, similarity, f"Similarity: {confidence:.1f}%"

# ============================================
# DATA MANAGEMENT
# ============================================
def load_users():
    if os.path.exists(REGISTERED_USERS_PATH):
        with open(REGISTERED_USERS_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(REGISTERED_USERS_PATH, 'w') as f:
        json.dump(users, f)

def save_user_mapping():
    with open(USER_ID_MAPPING_PATH, 'w') as f:
        json.dump({
            'user_id_to_class': user_id_to_class,
            'class_to_user_id': class_to_user_id
        }, f)

def save_embeddings():
    with open(USER_EMBEDDINGS_PATH, 'w') as f:
        json.dump(user_embeddings, f)

registered_users = load_users()

# Inventory
inventory = {
    '1': {'name': 'Coca Cola', 'price': 1.50, 'stock': 10},
    '2': {'name': 'Pepsi', 'price': 1.50, 'stock': 8},
    '3': {'name': 'Water', 'price': 1.00, 'stock': 15},
    '4': {'name': 'Chips', 'price': 2.00, 'stock': 12},
    '5': {'name': 'Candy Bar', 'price': 1.75, 'stock': 20},
    '6': {'name': 'Energy Drink', 'price': 2.50, 'stock': 5},
}

# User balances
user_balances = {}

# ============================================
# API ROUTES
# ============================================
@app.route('/')
def index():
    return send_from_directory('.', 'index_triple.html')

@app.route('/model_info')
def model_info():
    """Get information about loaded models"""
    return jsonify({
        'active_model': MODEL_TYPE,
        'models_available': {
            'siamese_pytorch': siamese_pytorch is not None,
            'modelA_classifier': modelA_classifier is not None,
            'modelB_siamese': modelB_siamese is not None
        },
        'features': {
            'auto_identification': modelA_classifier is not None,
            'embedding_based': modelB_siamese is not None and encoder_model is not None,
            'pytorch_verification': siamese_pytorch is not None
        },
        'num_registered_users': len(registered_users),
        'num_classes': len(label_classes) if label_classes is not None else 0
    })

@app.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.json
    user_id = data.get('user_id')
    image_data = data.get('image')
    initial_balance = data.get('balance', 10.0)
    
    if not user_id or not image_data:
        return jsonify({'success': False, 'message': 'Missing user ID or image'})
    
    if user_id in registered_users:
        return jsonify({'success': False, 'message': 'User already registered'})
    
    # Store the face image
    registered_users[user_id] = image_data
    user_balances[user_id] = initial_balance
    save_users(registered_users)
    
    # For Model A, assign class ID
    if modelA_classifier is not None:
        if label_classes is not None:
            # Find if user already in classes
            if user_id not in list(label_classes):
                print(f"⚠️  User {user_id} not in trained classes")
        if user_id not in user_id_to_class:
            new_class_id = len(user_id_to_class)
            user_id_to_class[user_id] = new_class_id
            class_to_user_id[str(new_class_id)] = user_id
            save_user_mapping()
    
    # For Model B, compute and store embedding
    if encoder_model is not None:
        try:
            img_array = process_image_keras(image_data)
            embedding = encoder_model.predict(img_array, verbose=0)[0]
            user_embeddings[user_id] = embedding.tolist()
            save_embeddings()
            print(f"✅ Stored embedding for {user_id}")
        except Exception as e:
            print(f"⚠️  Error computing embedding: {e}")
    
    return jsonify({
        'success': True,
        'message': f'User {user_id} registered successfully with {MODEL_TYPE}!',
        'balance': initial_balance,
        'model_type': MODEL_TYPE
    })

@app.route('/verify', methods=['POST'])
def verify():
    """Verify user identity"""
    data = request.json
    user_id = data.get('user_id')
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'message': 'Missing image'})
    
    try:
        if MODEL_TYPE == 'siamese_pytorch':
            if not user_id:
                return jsonify({'success': False, 'message': 'User ID required for PyTorch Siamese'})
            is_verified, metric, info = verify_pytorch_siamese(image_data, user_id)
            
        elif MODEL_TYPE == 'modelA':
            if user_id:
                is_verified, metric, info = verify_modelA(image_data, user_id)
            else:
                # Auto-identify
                _, metric, predicted_user = verify_modelA(image_data)
                is_verified = predicted_user is not None and metric > CONFIDENCE_THRESHOLD * 100
                user_id = predicted_user
                info = f"Auto-identified: {predicted_user}, Confidence: {metric:.1f}%"
                
        elif MODEL_TYPE == 'modelB':
            if not user_id:
                return jsonify({'success': False, 'message': 'User ID required for Model B'})
            is_verified, metric, info = verify_modelB(image_data, user_id)
        
        else:
            return jsonify({'success': False, 'message': 'No model available'})
        
        if is_verified and user_id:
            balance = user_balances.get(user_id, 0.0)
            return jsonify({
                'success': True,
                'message': f'Face verified! {info}',
                'metric': float(metric),
                'balance': balance,
                'user_id': user_id
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Verification failed! {info}',
                'metric': float(metric) if metric else 0
            })
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/identify', methods=['POST'])
def identify():
    """Auto-identify user (Model A only)"""
    if MODEL_TYPE != 'modelA':
        return jsonify({'success': False, 'message': 'Auto-identification only works with Model A'})
    
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'message': 'Missing image'})
    
    try:
        _, confidence, predicted_user = verify_modelA(image_data)
        
        if predicted_user and confidence > CONFIDENCE_THRESHOLD * 100:
            balance = user_balances.get(predicted_user, 0.0)
            return jsonify({
                'success': True,
                'user_id': predicted_user,
                'confidence': confidence,
                'balance': balance,
                'message': f'Identified as {predicted_user}'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Could not identify user (confidence: {confidence:.1f}%)',
                'confidence': confidence
            })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/purchase', methods=['POST'])
def purchase():
    """Process purchase with face payment"""
    data = request.json
    user_id = data.get('user_id')
    image_data = data.get('image')
    item_id = data.get('item_id')
    
    if not all([image_data, item_id]):
        return jsonify({'success': False, 'message': 'Missing required fields'})
    
    # Verify face
    try:
        if MODEL_TYPE == 'siamese_pytorch':
            if not user_id:
                return jsonify({'success': False, 'message': 'User ID required'})
            is_verified, metric, info = verify_pytorch_siamese(image_data, user_id)
            
        elif MODEL_TYPE == 'modelA':
            if user_id:
                is_verified, metric, info = verify_modelA(image_data, user_id)
            else:
                _, metric, user_id = verify_modelA(image_data)
                is_verified = user_id is not None and metric > CONFIDENCE_THRESHOLD * 100
                info = f"Auto-identified as {user_id}"
                
        elif MODEL_TYPE == 'modelB':
            if not user_id:
                return jsonify({'success': False, 'message': 'User ID required'})
            is_verified, metric, info = verify_modelB(image_data, user_id)
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Verification error: {str(e)}'})
    
    if not is_verified or not user_id:
        return jsonify({
            'success': False,
            'message': f'Face verification failed! {info}',
            'metric': float(metric) if metric else 0
        })
    
    # Check inventory
    if item_id not in inventory:
        return jsonify({'success': False, 'message': 'Invalid item'})
    
    item = inventory[item_id]
    if item['stock'] <= 0:
        return jsonify({'success': False, 'message': 'Item out of stock'})
    
    # Check balance
    user_balance = user_balances.get(user_id, 0.0)
    if user_balance < item['price']:
        return jsonify({
            'success': False,
            'message': f'Insufficient balance. Need ${item["price"]:.2f}, have ${user_balance:.2f}'
        })
    
    # Process purchase
    user_balances[user_id] -= item['price']
    inventory[item_id]['stock'] -= 1
    
    return jsonify({
        'success': True,
        'message': f'Purchase successful! Dispensing {item["name"]}',
        'item': item['name'],
        'price': item['price'],
        'remaining_balance': user_balances[user_id],
        'metric': float(metric) if metric else 0,
        'user_id': user_id
    })

@app.route('/balance/<user_id>')
def get_balance(user_id):
    balance = user_balances.get(user_id, 0.0)
    return jsonify({'user_id': user_id, 'balance': balance})

@app.route('/inventory')
def get_inventory():
    return jsonify(inventory)

@app.route('/add_balance', methods=['POST'])
def add_balance():
    data = request.json
    user_id = data.get('user_id')
    amount = data.get('amount', 0.0)
    
    if user_id not in registered_users:
        return jsonify({'success': False, 'message': 'User not registered'})
    
    user_balances[user_id] = user_balances.get(user_id, 0.0) + amount
    
    return jsonify({
        'success': True,
        'message': f'Added ${amount:.2f} to account',
        'new_balance': user_balances[user_id]
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🤖 Face Payment Vending Machine (Triple Model Support)")
    print("="*70)
    print(f"Active Model: {MODEL_TYPE.upper()}")
    print(f"PyTorch Siamese:      {'✅ Loaded' if siamese_pytorch else '❌ Not loaded'}")
    print(f"Model A (Classifier): {'✅ Loaded' if modelA_classifier else '❌ Not loaded'}")
    print(f"Model B (Siamese):    {'✅ Loaded' if modelB_siamese else '❌ Not loaded'}")
    if encoder_model:
        print(f"Encoder (for Model B): ✅ Loaded")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
