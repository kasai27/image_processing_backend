from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64

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

def process_image(image):
    # 画像処理のコード
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_image = process_image(image)

    # Base64エンコードしてフロントエンドに送信
    _, buffer = cv2.imencode('.jpg', processed_image)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    return  {"processed_image": base64_image}