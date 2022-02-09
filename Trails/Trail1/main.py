import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from landmarks import *
from model import load_model

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 1. New detection variables
sequence = []
sentence = []
threshold = 0.8
words = np.array(['hello', 'bye', 'thank you', 'sorry',
                 'yes', 'no', 'please', 'house'])


# Testing model
model = load_model()


cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
#         sequence.insert(0,keypoints)
#         sequence = sequence[:30]
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0][:-1]
            # if np.max(res) > 0.75 :
            print(res)
            print(words[np.argmax(res)])

        # 3. Viz logic
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if words[np.argmax(res)] != sentence[-1]:
                        sentence.append(words[np.argmax(res)])
                else:
                    sentence.append(words[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Viz probabilities
            # image = prob_viz(res, actions, image, colors)

#         cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        # cv2.putText(image, words[np.argmax(res)], (3, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
