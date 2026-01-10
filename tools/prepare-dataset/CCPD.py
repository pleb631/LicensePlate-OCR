from pathlib import Path
from itertools import chain
from typing import  Union
import pickle as pkl
import numpy as np
import cv2
import tqdm
import mxnet as mx
import random

def parse_ccpd_label(image_path: Union[str, Path]):
    image_path = Path(image_path)
    provinces = '皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学O'
    alphabets = 'ABCDEFGHJKLMNPQRSTUVWXYZO'
    ads = 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789O'
    area, title_degree, box, lamdmark, plate, brightness, blurriness = image_path.stem.split('-')
    plate_parts = plate.split('_')
    return {
        'area': float(f"0.{area}"),
        'title_degree': tuple(map(float, title_degree.split('_'))),
        'box': tuple(map(int, chain.from_iterable(map(lambda x: x.split('&'), box.split('_'))))),
        'landmark': np.fromiter((map(int, chain.from_iterable(map(lambda x: x.split('&'), lamdmark.split('_'))))), dtype=np.int32).reshape(-1, 2),
        'plate': f"{provinces[int(plate_parts[0])]}{alphabets[int(plate_parts[1])]}{''.join(map(lambda x: ads[int(x)], plate_parts[2:]))}",
        'brightness': float(brightness),
        'blurriness': float(blurriness),
    }

def xyxy2xywh(xyxy):
    x_center = (xyxy[0] + xyxy[2]) // 2
    y_center = (xyxy[1] + xyxy[3]) // 2
    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]
    xywh_center = [x_center, y_center, width, height]

    return xywh_center

def expand_box(xyxy, ratio,w,h):
    if isinstance(ratio, float) or isinstance(ratio, int):
        ratio = [ratio, ratio]
    xywh = xyxy2xywh(xyxy)
    new_w, new_h = xywh[2] * ratio[0], xywh[3] * ratio[1]
    x1, y1 = xywh[0] - new_w // 2, xywh[1] - new_h // 2
    x2, y2 = x1+new_w, y1 +new_h
    x1 = np.clip(x1,0,w)
    x2 = np.clip(x2,0,w)
    y1 = np.clip(y1,0,h)
    y2 = np.clip(y2,0,h)
    return np.array([x1, y1, x2, y2])


def containExtraAlphabet(line):
    if line[0] not in '京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新':
        return True
    for alphabet in line[1:]:
        if alphabet not in '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ':
            return True
    return False
def process(path,root,dst):
    anno = parse_ccpd_label(path)
    
    if anno["plate"][0]=='皖':
        if random.random()>0.04:
            return None
    if containExtraAlphabet(anno['plate']):
        return None
    im = cv2.imread(str(path))
    h,w=im.shape[:2]
    bbox = anno['box']
    landmark = anno['landmark']
    landmark = landmark[[2,3,0,1]]
    new_bbox = expand_box(bbox,[1.1,1.4],w,h)
    new_bbox = list(map(int,new_bbox))
    crop_landmark = landmark-new_bbox[:2]
    crop_img = im[new_bbox[1]:new_bbox[3],new_bbox[0]:new_bbox[2]].copy()
    crop_bbox = np.array(bbox).reshape(-1,2)-np.array(new_bbox[:2]).reshape(-1,2)
    h,w=crop_img.shape[:2]
    crop_h,crop_w = crop_bbox[1,1]-crop_bbox[0,1],crop_bbox[1,0]-crop_bbox[0,0]
    ih,iw=32,96
    if crop_h>ih and crop_w>iw:
        ratio = 1+(min(crop_h/ih,crop_w/iw)-1)/1.7
        resize_box = crop_bbox/[ratio,ratio]
        resize_landmark = crop_landmark/[ratio,ratio]
        resize_img = cv2.resize(crop_img,(int(w/ratio),int(h//ratio)))

        return {
        'title_degree': anno['title_degree'],
        'box': np.round(resize_box).astype(np.int16).reshape(-1).tolist(),
        'landmark': np.round(resize_landmark).astype(np.int16).reshape(-1).tolist(),
        'plate': anno['plate'],
        'img':cv2.imencode(".png", resize_img)[1]
        }
    # cv2.rectangle(im,[new_bbox[0],new_bbox[1]],[new_bbox[2],new_bbox[3]],[255,0,0],2)
    # for point in new_landmark:
    #     cv2.circle(crop_img,point,5,[255,0,0],-1)
    return {
        'title_degree': anno['title_degree'],
        'box': np.round(crop_bbox).astype(np.int16).reshape(-1).tolist(),
        'landmark': np.round(crop_landmark).astype(np.int16).reshape(-1).tolist(),
        'plate': anno['plate'],
        'img':cv2.imencode(".png", crop_img)[1]
    }


if __name__ == '__main__':
    root = r"path/to/CCPD2020/"
    root1 = r"path/to/CCPD2019/"
    dst = r'dataset/CCPD/'
    Path(dst).mkdir(exist_ok=True,parents=True)
    write_record = mx.recordio.MXIndexedRecordIO(dst+'CCPD.idx',dst+'CCPD.rec','w')
    items = list(Path(root).rglob("*.jpg"))+list(Path(root1).rglob("*.jpg"))
    #items = Path(root1).rglob("*.jpg")
    i=0
    sum=0
    for item in tqdm.tqdm(items):
        try:
            anno=process(item,Path(root),Path(dst))
        except:
            continue
        if anno is None:
            # print(f'pass {str(item)}')
            continue

        item_byte = pkl.dumps(anno)
        sum+=len(item_byte)
        write_record.write_idx(i,item_byte)
        i+=1
    print(f'{sum/1024/1024} mb')
    write_record.close()
    record = mx.recordio.MXIndexedRecordIO(dst+'CCPD.idx',dst+'CCPD.rec','r')
    item_byte = record.read_idx(1)
    item = pkl.loads(item_byte)
    print(item['plate'])