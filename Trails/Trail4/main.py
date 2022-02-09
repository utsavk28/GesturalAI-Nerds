import os
import numpy as np
import cv2
from models import le, model
from utils import process_image_data

# labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
#           15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}

cap = cv2.VideoCapture(0)
while True:
    isTrue, frame = cap.read()

    if not isTrue:
        break

    img = frame[:200, :200]
    # print(img.shape)
    feat = process_image_data(img)
    # print(feat.shape)
    res = model.predict(feat)
    # print(res.shape)
    n = np.argmax(res)
    # print(res[n], le.inverse_transform([res[n]]))
    # print(res[n] )
    # print(frame.shape)
    h, w, c = frame.shape
    cv2.rectangle(frame, (0, 0), (200, 200), (225, 0, 255), 2)
    cv2.putText(frame, le.inverse_transform([res[n]])[0], (w//2, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('OpenCV Feed', frame)

    # Break gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
