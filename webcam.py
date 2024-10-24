from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import img_to_array
from keras._tf_keras.keras.models import load_model
import mediapipe as mp
import numpy as np
import cv2
import cvlib as cv

# Loads model created previously
# gender_detection.model
model = load_model(r'/Users/vasudevnair113/Downloads/archive/sign_language_detection.keras')

# Opens up the webcam
# Index = camera you want to use
webcam = cv2.VideoCapture(0)

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Loops through the frames coming from the webcam
while webcam.isOpened():
    # Looks at the current frame
    status, frame = webcam.read()
    # If the frame was not captured, then this iteration of the for loop is skipped
    # This means only successfully captured frames are processed further
    if not status:
        continue
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    result = hands.process(rgb_frame)
    
    # Checks detected hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get bounding box coordinates
            h, w, c = frame.shape # c = number of color channels (3 for RGB)
            # Converting normalized landmark coordinates to pixel coordinates
            x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
            y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
            x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
            y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)
            
            # Draw bounding box
            # (0, 255, 0) = Draws a green box
            # 2 = Border thickness
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Crop hand out from frame and convert to numpy array
            hand_crop = np.copy(frame[y_min:y_max, x_min:x_max])
            
            # Checks to see if cropped hand is too small
            if hand_crop.shape[0] < 10 or hand_crop.shape[1] < 10:
                continue
            
            # Data preprocessing of hand before feeding it to model
            hand_crop = cv2.resize(hand_crop, (96, 96)) # Dimensions the model is trained on
            hand_crop = hand_crop.astype('float') / 255.0 # Changes to 0-1 scale, we did this in training file
            hand_crop = img_to_array(hand_crop)
            hand_crop = np.expand_dims(hand_crop, axis=0) # Adds extra dimension to array
            
            # Passing hand through our model
            conf = model.predict(hand_crop)[0]
            
            # Argmax returns index of the largest element in an array
            idx = np.argmax(conf)
            
            label = classes[idx]
            # Formatting into string (letter and confidence level)
            label = '{}: {:.2f}%'.format(label, conf[idx] * 100)
            
            # Calculates vertical position for label text
            Y = y_min - 10 if y_min - 10 > 10 else y_min + 10
            
            # Writing onto the frame
            # 0.7 = Font scale
            # (0, 255, 0) = Green text
            # 2 = Thickness of text
            cv2.putText(frame, label, (x_min, Y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Displays output on screen
    cv2.imshow('Sign Language Detection', frame)
    
    # If user presses 's' then the loop breaks
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# Once loop is broken, this closes webcam
webcam.release()
cv2.destroyAllWindows()