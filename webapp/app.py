from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
from models import resnet_model, mobilenet_model

app = Flask(__name__, static_url_path='/static')
model = resnet_model
labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
          15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        isTrue, frame = cap.read()

        if not isTrue:
            break

        img = frame[:200, :200]
        img = cv2.resize(img, (100, 100,))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img).reshape((1, 100, 100, 3))
        res = model.predict(img)[0]
        print(res)
        n = np.argmax(res)
        print(res[n], labels[n])
        print(frame.shape)
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
    while True:

        # read the camera frame
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
