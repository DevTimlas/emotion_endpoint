from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import uvicorn

from tensorflow.keras.models import model_from_json

app = FastAPI()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model_json_file = 'model.json'
model_weights_file = 'model_weights.h5'
with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_file)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # listen for connections
    await websocket.accept()

    try:
        while True:
            contents = await websocket.receive_bytes()
            arr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                fc = gray[y:y + h, x:x + w]
                roi = cv2.resize(fc, (48, 48))
                pred = loaded_model.predict(roi[np.newaxis, :, :, np.newaxis])
                text_idx = np.argmax(pred)
                text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
                text = text_list[text_idx]
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.imshow('frame', frame)
            cv2.waitKey(1)

    except WebSocketDisconnect:
        cv2.destroyWindow("frame")
        print("Client disconnected")
