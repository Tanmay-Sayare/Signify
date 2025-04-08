import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tqdm import tqdm

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_hand_features(image_path):
    """Extract hand landmarks from an image and convert to features."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Get the first hand's landmarks
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # Extract coordinates
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    
    return features

def load_dataset(data_dir):
    """Load and process the dataset."""
    X = []
    y = []
    
    # Get all gesture directories
    gesture_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for gesture_id in tqdm(gesture_dirs, desc="Processing gestures"):
        gesture_dir = os.path.join(data_dir, gesture_id)
        
        # Get all images in the gesture directory
        image_files = [f for f in os.listdir(gesture_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for image_file in tqdm(image_files, desc=f"Processing images for gesture {gesture_id}", leave=False):
            image_path = os.path.join(gesture_dir, image_file)
            features = extract_hand_features(image_path)
            
            if features is not None:
                X.append(features)
                y.append(int(gesture_id))
    
    return np.array(X), np.array(y)

def create_model(input_shape, num_classes):
    """Create a neural network model."""
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Load and process the dataset
    print("Loading dataset...")
    X, y = load_dataset('data')
    
    # Convert labels to one-hot encoding
    num_classes = len(np.unique(y))
    y = to_categorical(y, num_classes=num_classes)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    print("Creating model...")
    model = create_model(input_shape=X.shape[1], num_classes=num_classes)
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        'sign_language_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Train the model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint],
        verbose=1
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    print("Model saved as 'sign_language_model.keras'")

if __name__ == "__main__":
    main() 