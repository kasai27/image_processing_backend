from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from library import fragile
from library import detection

app = FastAPI()

origins = {
    "http://localhost:3000",
}


app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers= ["*"],
)

# グレースケール化
def process_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# ラプラシアン処理（エッジ）
def process_laplacian(image):
    laplacian_image = cv2.Laplacian(image, -1)
    return laplacian_image

# グレーススケール化
@app.post("/gray/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    processed_image = process_grayscale(image)

    # Base64エンコードしてフロントエンドに送信
    _, buffer = cv2.imencode('.jpg', processed_image)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    return  {"processed_image": base64_image}

# エッジ処理（ラプラシアン処理）
@app.post("/laplacian/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_image = process_laplacian(image)

    # Base64エンコードしてフロントエンドに送信
    _, buffer = cv2.imencode('.jpg', processed_image)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    return  {"processed_image": base64_image}

# 電子透かし処理
@app.post("/watermark/")
async def process_data(file: UploadFile = File(...), text: str = Form(...)):
    contents = await file.read()
    text = {text}

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_image = fragile.create_fragile_image(image, text)
    #result_text = detection.detection(processed_image)
    #print(result_text)

    # Base64エンコードしてフロントエンドに送信
    _, buffer = cv2.imencode('.jpg', processed_image)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    return  {"processed_image": base64_image}
