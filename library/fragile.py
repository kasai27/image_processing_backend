### 画像の離散コサイン変換 (Y成分のみ) ###
import cv2
import numpy as np
from scipy.fftpack import dct, idct
import random
import math
from library import str_bin_change

# 2次元離散コサイン変換 (DCT)
def dct_2d(img):
    return dct(dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')


# 2次元逆離散コサイン変換 (IDCT)
def idct_2d(img_dct):
    return idct(idct(img_dct, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def create_fragile_image(img, text):
    # parameters
    watermark_length = 1024
    seed = 10
    step = 20
    
    # 画像の色空間の変換
    img = img.astype(np.float32)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # 二次元DCT
    img_dct = dct_2d(img_yuv[:,:,0])

    data = str_bin_change.str_to_bin(text)
    index_list = []
    s_t = []

    # 透かし情報埋め込み位置を決める
    for i in range(1, len(img_dct)):
        for j in range(51):
            index_list.append([i, j])
    random.seed(seed)
    sampling_list = random.sample(index_list, watermark_length)

    for n in range(watermark_length):
        i, j = sampling_list[n]
        s_t.append(img_dct[i, j])

    #一次元DCT
    S_t = dct(s_t, axis=0, norm='ortho')

    #QIM法による埋め込み
    count = 0

    for n in range(watermark_length):
        if math.floor(S_t[n]/step)%2 == int(data[count]):
            S_t[n] = step * math.floor(S_t[n]/step)
        else:
            S_t[n] = step * math.floor(S_t[n]/step + 1)
        count += 1

    #一次元逆DCT
    s_t = idct(S_t, axis=0 , norm='ortho')

    for n in range(watermark_length):
        i, j = sampling_list[n]
        img_dct[i, j] = s_t[n]

    # 逆二次元DCT
    img_inv = idct_2d(img_dct)

    # 色空間をYUV表示系からRGB表示系に変換
    img_yuv[:,:,0] = img_inv
    img_bgr_inv = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_bgr_inv

