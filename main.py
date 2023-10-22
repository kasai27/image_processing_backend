from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.responses import JSONResponse
import cv2
import numpy as np
import base64
from library import fragile
from library import detection

app = FastAPI()

origins = {
    "http://localhost:3000",
    "https://image-processing-frontend.vercel.app/detection"
}


app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers= ["*"],
)

# グレースケール化関数
def process_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# エッジ処理関数
def ege_processing(image, fileter_type):
    if fileter_type == "Sobel":
        ege_processed_image = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    elif fileter_type == "Laplacian":
        ege_processed_image = cv2.Laplacian(image, cv2.CV_32F)
    elif fileter_type =="Canny":
        ege_processed_image = cv2.Canny(image, 100, 200)
    return ege_processed_image




# グレーススケール化(API)
@app.post("/gray/")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 画像処理
        processed_image = process_grayscale(image)

        retval, buffer = cv2.imencode(".jpg", processed_image)
        image_bytes = buffer.tobytes()

        def generate():
            yield image_bytes

        return StreamingResponse(generate(), media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# エッジ処理(API)
@app.post("/edge/")
async def upload_image(file: UploadFile = File(...), filter_type: str = Form(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        processed_image = ege_processing(image, filter_type)

        retval, buffer = cv2.imencode(".jpg", processed_image)
        image_bytes = buffer.tobytes()

        def generate():
            yield image_bytes

        return StreamingResponse(generate(), media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

# 電子透かし処理(API)
@app.post("/watermark/")
async def process_data(file: UploadFile = File(...), text: str = Form(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        processed_image = fragile.create_fragile_image(image, text)

        retval, buffer = cv2.imencode(".png", processed_image)
        image_bytes = buffer.tobytes()

        def generate():
            yield image_bytes
        
        return StreamingResponse(generate(), media_type="image/png")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# 電子透かし抽出処理(API)
@app.post("/detection/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result_text = detection.detection(image)
    result_text = result_text.rstrip("\0")
    print("result:" + result_text)

    return  {"result_text": result_text}

# 顔検出処理(API)
@app.post("/face_detection/")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        cascade_path = "./library/haarcascade_frontalface_alt.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        facerect = cascade.detectMultiScale(image)

        if len(facerect) > 0:
            for rect in facerect:
                cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0, 255), thickness=2)
        else:
            print('no face')

        retval, buffer = cv2.imencode(".jpg", image)
        image_bytes = buffer.tobytes()

        def generate():
            yield image_bytes

        return StreamingResponse(generate(), media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
