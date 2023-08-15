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

# エッジ処理
def ege_processing(image, fileter_type):
    if fileter_type == "Sobel":
        ege_processed_image = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    elif fileter_type == "Laplacian":
        ege_processed_image = cv2.Laplacian(image, cv2.CV_32F)
    elif fileter_type =="Canny":
        ege_processed_image = cv2.Canny(image, 100, 200)
    return ege_processed_image

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
@app.post("/edge/")
async def upload_image(file: UploadFile = File(...), filter_type: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_image = ege_processing(image, filter_type)

    # Base64エンコードしてフロントエンドに送信
    _, buffer = cv2.imencode('.jpg', processed_image)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    return  {"processed_image": base64_image}

# 電子透かし処理
@app.post("/watermark/")
async def process_data(file: UploadFile = File(...), text: str = Form(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_image = fragile.create_fragile_image(image, text)
    result_s = detection.detection(processed_image)
    print(result_s)
    output_path = "download_files/" + file.filename
    cv2.imwrite(output_path, processed_image)

    # Base64エンコードしてフロントエンドに送信
    _, buffer = cv2.imencode('.jpg', processed_image)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    return  {"processed_image": base64_image}

# 電子透かし抽出処理
@app.post("/detection/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result_text = detection.detection(image)
    result_text = result_text.rstrip("\0")
    print("result:" + result_text)

    return  {"result_text": result_text}