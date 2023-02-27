# Import necessary libraries
import cv2
import numpy as np
#pip install deepface
from deepface import DeepFace

# Load DEEPFACE model
model = DeepFace.build_model('Emotion')

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Resize frame
    resized_frame = cv2.resize(frame, (48, 48), interpolation = cv2.INTER_AREA)
    
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    
    # Preprocess the image for DEEPFACE
    img = gray_frame.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    
    # Predict emotions using DEEPFACE
    preds = model.predict(img)
    emotion_idx = np.argmax(preds)
    emotion = emotion_labels[emotion_idx]
    
    # Draw rectangle around face and label with predicted emotion
    cv2.rectangle(frame, (0, 0), (200, 30), (0, 0, 0), -1)
    cv2.putText(frame, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
