import numpy as np
import numba as nu
import cv2
import time


def si(cur):
    return None


def he(cur):
    return None


start_time = time.time()

dataraw = np.fromfile('3.data', dtype=np.ushort())
data12 = np.floor(dataraw / 65535 * 4095)  #переводим в 12 бит
# dataCLAHE = []
data12 = data12.astype(np.ushort())


@nu.jit
def chist_sub(data, mem):
    hist_arr = np.bincount(data.flatten(), minlength=4096)
    hist_arr = cut(hist_arr, mem)
    hist_arr = hist_arr / np.sum(hist_arr)
    chist_arr = np.cumsum(hist_arr)
    transform_map = np.floor(65535 * chist_arr).astype(np.ushort())
    return transform_map


@nu.jit
def cut(arr, height):
    summ = 0
    for i in range(4096):
        if arr[i] >= height:
            summ += arr[i] - height
            arr[i] = height
    summ = summ / 4096
    for i in range(4096):
        arr[i] += summ
    return arr


@nu.jit
def clahe(dat, ss, frames=250):
    root_window = 'CLAHE'
    cv2.namedWindow(root_window)
    cv2.createTrackbar('top', root_window, 1, 50, he)
    cv2.createTrackbar('size', root_window, 0, 2, si)
    cv2.setTrackbarPos('size', root_window, 1)

    #dataCLAHE = []
    data = np.reshape(dat, (250, 512, 640))

    """Сегментирование кадра и подсчет нор. кум. гистограммы"""
    for fr in range(frames):
        height = cv2.getTrackbarPos('top', root_window)
        sb = cv2.getTrackbarPos('size', root_window)
        ss = 2 ** (5 + sb)
        # ss Размер сегмента
        sc = ss // 2  # Центр сегмента
        x = 512 // ss
        y = 640 // ss

        frame = data[fr]
        subs_ch = np.empty((x, y), dtype=np.ndarray)
        subs = np.empty((x, y), dtype=np.ndarray)
        for i in range(x):
            for j in range(y):
                sub = frame[i * ss:i * ss + ss, j * ss:j * ss + ss]
                subs_ch[i, j] = chist_sub(sub, height)
                subs[i, j] = sub

        ###Интерполяция### 
        for i in range(sc, 512 - sc):
            for j in range(sc, 640 - sc):
                frame[i, j] = (1 / (ss ** 2)) * \
                              (subs_ch[(i - sc) // ss, (j - sc) // ss][frame[i, j]] * (ss - (i - sc) % ss) * (
                                          ss - (j - sc) % ss) +\
                               subs_ch[(i - sc) // ss, (j - sc) // ss + 1][frame[i, j]] * (ss - (i - sc) % ss) * (
                                           (j - sc) % ss) +\
                               subs_ch[(i - sc) // ss + 1, (j - sc) // ss][frame[i, j]] * ((i - sc) % ss) * (
                                           ss - (j - sc) % ss) +\
                               subs_ch[(i - sc) // ss + 1, (j - sc) // ss + 1][frame[i, j]] * ((i - sc) % ss) * ((j - sc) % ss))

        ###Грани### 
        for i in range(sc, 512 - sc):
            for j in range(0, sc):
                frame[i, j] = (1 / ss) * \
                              (subs_ch[(i - sc) // ss, 0][frame[i, j]] * (ss - (i - sc) % ss) +\
                               subs_ch[(i - sc) // ss + 1, 0][frame[i, j]] * ((i - sc) % ss))

        for i in range(sc, 512 - sc):
            for j in range(640 - sc, 640):
                frame[i, j] = (1 / ss) * \
                              (subs_ch[(i - sc) // ss, y - 1][frame[i, j]] * (ss - (i - sc) % ss) +\
                               subs_ch[(i - sc) // ss + 1, y - 1][frame[i, j]] * ((i - sc) % ss))

        for i in range(0, sc):
            for j in range(sc, 640 - sc):
                frame[i, j] = (1 / ss) * \
                              (subs_ch[0, (j - sc) // ss][frame[i, j]] * (ss - (j - sc) % ss) +\
                               subs_ch[0, (j - sc) // ss + 1][frame[i, j]] * ((j - sc) % ss))

        for i in range(512 - sc, 512):
            for j in range(sc, 640 - sc):
                frame[i, j] = (1 / ss) * \
                              (subs_ch[x - 1, (j - sc) // ss][frame[i, j]] * (ss - (j - sc) % ss) +\
                               subs_ch[x - 1, (j - sc) // ss + 1][frame[i, j]] * ((j - sc) % ss))

        ###Углы###
        for i in range(0, sc):
            for j in range(0, sc):
                frame[i, j] = subs_ch[0, 0][frame[i, j]]

        for i in range(512 - sc, 512):
            for j in range(0, sc):
                frame[i, j] = subs_ch[x - 1, 0][frame[i, j]]

        for i in range(0, sc):
            for j in range(640 - sc, 640):
                frame[i, j] = subs_ch[0, y - 1][frame[i, j]]

        for i in range(512 - sc, 512):
            for j in range(640 - sc, 640):
                frame[i, j] = subs_ch[x - 1, y - 1][frame[i, j]]

        #dataCLAHE.append(frame)

        cv2.imshow('CLAHE', frame)
        cv2.waitKey(1)
    return #dataCLAHE

""""""
clahe(data12, 64)
#Vid_arr = clahe(data12, 64)
#np.save('videoCLAHE', Vid_arr)
print("---%s seconds ---" % (time.time() - start_time))