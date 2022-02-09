import os
import numpy as np
import cv2
from models import resnet_model, model1, model2, model_bw
import numpy as np
import cv2
minValue = 70

model = model_bw
labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
          15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}

cap = cv2.VideoCapture(0)
while True:
    isTrue, frame = cap.read()

    if not isTrue:
        break

    img = frame[:200, :200]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(
        th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    res = cv2.resize(res,(128,128))
    img = np.array(res).reshape((1, 128, 128))/255
    print(img)
    res = model.predict(img)[0]
    # print(res)
    n = np.argmax(res)
    print(res[n], labels[n])
    # print(frame.shape)
    h, w, c = frame.shape

    cv2.rectangle(frame, (0, 0), (200, 200), (225, 0, 255), 2)
    cv2.putText(frame, labels[n], (w//2, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('OpenCV Feed', frame)

    # Break gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
