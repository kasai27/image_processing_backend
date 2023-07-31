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

def detection(img):
    # parameters
    watermark_length = 1024
    seed = 10
    step = 20

    # 画像の色空間の変換
    img = img.astype(np.float32)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # 二次元DCT
    img_dct = dct_2d(img_yuv[:,:,0])

    #埋め込んだ位置を検出
    index_list = []
    s_t = []

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

    #埋め込み情報の検出
    data_w = []
    for n in range(watermark_length):
        data_w.append(math.floor(S_t[n]/step+0.5)%2)
    result = "".join(map(str, data_w))
    result_s = str_bin_change.bin_to_str(result)
    
    return result_s