"""
import websockets
import asyncio
import cv2

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

async def main():
    # Connect to the server
    async with websockets.connect('ws://localhost:5000/ws') as ws:
         while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.png', frame)
                await ws.send(buffer.tobytes())

# Start the connection
asyncio.run(main())
"""

import cv2
import websockets
import asyncio

async def send_frames(websocket):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        _, img_encoded = cv2.imencode('.jpg', frame)
        await websocket.send(img_encoded.tobytes())

async def main():
    async with websockets.connect('ws://localhost:5000/ws') as websocket:
        await send_frames(websocket)

if __name__ == '__main__':
    asyncio.run(main())

