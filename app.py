import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PIL import Image

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('sign_language_model.keras')

def extract_hand_features(frame):
    """Extract hand landmarks from a frame and convert to features."""
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = hands.process(frame_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Get the first hand's landmarks
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # Extract coordinates
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    
    return np.array(features)

def main():
    st.title("Real-time Sign Language Recognition")
    st.write("Show your hand gesture to the webcam for recognition")
    
    # Load the model
    model = load_model()
    
    # Create a placeholder for the video feed
    frame_placeholder = st.empty()
    
    # Add a start/stop button
    start_button = st.button("Start/Stop Camera")
    
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    
    if start_button:
        st.session_state.is_running = not st.session_state.is_running
    
    if st.session_state.is_running:
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        while st.session_state.is_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break
            
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Extract hand features
            features = extract_hand_features(frame)
            
            if features is not None:
                # Make prediction
                prediction = model.predict(np.expand_dims(features, axis=0))
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]
                
                # Draw prediction on frame
                cv2.putText(frame, f"Gesture: {predicted_class}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert the frame to RGB for display in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            frame_placeholder.image(frame_rgb, channels="RGB")
            
            # Add a small delay to reduce CPU usage
            cv2.waitKey(1)
        
        # Release the webcam when done
        cap.release()

if __name__ == "__main__":
    main() 