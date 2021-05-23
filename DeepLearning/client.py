import asyncio
import time
import socketio
import random
import models.yolov3 as yolo
import numpy as np
import time


loop = asyncio.get_event_loop()
sio = socketio.AsyncClient()
start_timer = None

async def send_ping(latency= 0):
    global start_timer
    start_timer = time.time()
    package, boxes, image = engine.inference(inputs, model, sess, cap)
    await sio.emit("ping_from_client", {"value":list(np.array(boxes).shape), "package" : package, "latency": latency, "image": image})


@sio.event
async def connect():
    print('connected to server')
    await send_ping()
@sio.event
async def pong_from_server(data ):
    global start_timer
    latency = time.time() - start_timer
    if sio.connected:
        await send_ping(latency)   
async def start_server():
    await sio.connect('http://localhost:3000')
    await sio.wait()  
if __name__ == '__main__':
    #Change camara_feed parameter
    # engine = yolo.InferenceEngine(camera_feed = "rtsp://admin:admin0000@192.168.1.7:554/cam/realmonitor?channel=2&subtype=1")
    engine = yolo.InferenceEngine(camera_feed = 0)
    inputs, model, sess, cap , error= engine.load()
    if not error:
        loop.run_until_complete(start_server())


#Notes
#rtsp url: "rtsp://admin:admin0000@192.168.1.7:554/cam/realmonitor?channel=2&subtype=1"