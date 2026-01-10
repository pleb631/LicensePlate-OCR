import matplotlib.pyplot as plt
import numpy as np
import cv2


def save_txt(txt_path, info, mode='w'):
    '''保存txt文件

    Args:
        txt_path: str, txt文件路径
        info: list, txt文件内容
        mode: str, 'w'代表覆盖写；'a'代表追加写
    '''
#    os.makedirs(os.path.split(txt_path)[0], exist_ok=True)
    
    txt_file = open(txt_path, mode,encoding="UTF-8")
    for line in info:
        txt_file.write(line + '\n')
    txt_file.close()
    
def imread(path) -> np.ndarray:
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    return bgrImage


def grid_plot_image(images, labels, cols=4, **kwargs):
    n = len(images)
    rows = (n + cols - 1) // cols
    if kwargs:
        plt.figure(**kwargs)
    for index in range(n):
        plt.subplot(rows, cols, index + 1)
        plt.title(str(labels[index]))
        plt.axis("off")
        plt.imshow(images[index])
    plt.show()



def xyxy2xywh_center(xyxy):
    '''[xmin, ymin, xmax, ymax]转为[x_center, y_center, w, h]

    Args:
        xyxy: list, 格式[xmin, ymin, xmax, ymax]

    Returns:
        xywh_center: list, 格式[x_center, y_center, w, h]
    '''
    x_center = int((xyxy[0] + xyxy[2]) / 2)
    y_center = int((xyxy[1] + xyxy[3]) / 2)
    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]
    xywh_center = [x_center, y_center, width, height]

    return xywh_center

def xywh_center2xyxy(xywh):
    '''[x_center, y_center, w, h]转为[xmin, ymin, xmax, ymax]

    Args:
        xywh: list, 格式[x_center, y_center, w, h]

    Returns:
        xyxy: list, 格式[xmin, ymin, xmax, ymax]
    '''
    xmin = int(xywh[0] - xywh[2] / 2)
    ymin = int(xywh[1] - xywh[3] / 2)
    xmax = int(xywh[0] + xywh[2] / 2)
    ymax = int(xywh[1] + xywh[3] / 2)
    xyxy = [xmin, ymin, xmax, ymax]

    return xyxy
