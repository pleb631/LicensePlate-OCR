from pathlib import Path
from typing import Union
import pickle as pkl
import numpy as np
import cv2
import tqdm
import mxnet as mx



def containExtraAlphabet(line):
    if line[0] not in '京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新':
        
        return True
    for alphabet in line[1:]:
        if alphabet not in '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ':
            return True
    
    return False

def quadrilateral_points2left_top_first_quadrilateral(quadrilateral_points, mode='left_top_euclidean'):
    '''凸四边形[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]。以左上角点为起始点，按顺时针排序，转为新四边形[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        确定左上角点，再与剩余三个点连线。连线后，如果存在一个点在线上方，一个点在线下方。那么线上的点为右上，线下的点为左下，连接点为右下
        
        定理：向量a×向量b（×为向量叉乘），若结果小于0，表示向量b在向量a的顺时针方向；若结果大于0，表示向量b在向量a的逆时针方向；若等于0，表示向量a与向量b平行
        跨立实验：如果线段CD的两个端点C和D，与另一条线段的一个端点（A或B，只能是其中一个）连成的向量，与向量AB做叉乘，
            若结果异号，表示C和D分别在直线AB的两边，若结果同号，则表示CD两点都在AB的一边，则肯定不相交。
        https://blog.csdn.net/qq_40733911/article/details/99121758

    Args:
        quadrilateral_points: list, 原始四边形。格式[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        mode: str, 若为'left_top_euclidean'，则认为离原点欧氏距离最小的为左上角；若为'left_top_position'，则先找最上的两个点，再找最左的点作为左上角（比如行人框）

    Returns:
        left_top_first_points: list, 左上角为起始点，顺时针排序的新四边形。格式[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    '''
    quadrilateral_points = np.array(quadrilateral_points)

    if mode == 'left_top_euclidean':
        # 认为离原点欧氏距离最小的为左上角
        origin_dist = np.sqrt(np.sum(np.square(quadrilateral_points),1))
        left_top_index = np.argmin(origin_dist)
    elif mode == 'left_top_position':
        # 先找最上的两个点，再找最左的点作为左上角（比如行人框）
        quadrilateral_points_for_lt = quadrilateral_points.copy()
        ymin2_index = np.argpartition(quadrilateral_points_for_lt[:,1], -2)
        ymin2_index = ymin2_index[-2:]
        quadrilateral_points_for_lt[ymin2_index] = [1e6, 1e6]
        left_top_index = np.argmin(quadrilateral_points_for_lt[:,0])
    
    # # 如果本来第一位就是左上角的点
    # if left_top_index == 0:
    #     return quadrilateral_points

    for index in range(quadrilateral_points.shape[0]):
        if index == left_top_index:
            continue
        remaining_points = list(range(quadrilateral_points.shape[0]))
        remaining_points.remove(index)
        remaining_points.remove(left_top_index)

        # 左上角点为A点
        line_AB = np.array((quadrilateral_points[index][0] - quadrilateral_points[left_top_index][0], quadrilateral_points[index][1] - quadrilateral_points[left_top_index][1]))
        line_AC = np.array((quadrilateral_points[remaining_points[0]][0] - quadrilateral_points[left_top_index][0], quadrilateral_points[remaining_points[0]][1] - quadrilateral_points[left_top_index][1]))
        line_AD = np.array((quadrilateral_points[remaining_points[1]][0] - quadrilateral_points[left_top_index][0], quadrilateral_points[remaining_points[1]][1] - quadrilateral_points[left_top_index][1]))
        cross_ABC = np.cross(line_AB, line_AC)
        cross_ABD = np.cross(line_AB, line_AD)
        # 如果两线段叉乘异号，代表两线段相交
        if cross_ABC * cross_ABD < 0:
            # 叉积小于0，代表AC在AB顺时针方向，因为图像坐标系原点在左上角，顺逆时针与人看的时候相反，所以C为右上角点
            if cross_ABC < 0:
                right_top_index = remaining_points[0]
                right_bottom_index = index
                left_bottom_index = remaining_points[1]
            else:
                right_top_index = remaining_points[1]
                right_bottom_index = index
                left_bottom_index = remaining_points[0]

    left_top_first_points = quadrilateral_points[[left_top_index, right_top_index, right_bottom_index, left_bottom_index], :]

    return left_top_first_points


def parse_crpd_label(image_path: Union[str, Path]):
    image_path = Path(image_path)
    label_path = image_path.parent.parent/"labels"/(image_path.stem+'.txt')
    txt_file = open(label_path, "r", encoding='utf-8-sig')
    out = []
    for item in txt_file.readlines():
        item = item.strip().split()
        if item[-2]=='2':
            continue
        plate = item[-1]
        if containExtraAlphabet(plate):
            print(plate)
            continue
        landmark=list(map(int,item[:-2]))
        land = np.array(landmark).reshape(4,2)
        bbox = np.array([np.min(land,axis=0),np.max(land,axis=0)]).reshape(-1)
        
        anno = {
        'box': bbox,
        'landmark': land,
        'plate': plate,}
        out.append(anno)
        break
    return out

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


def process_single(path,anno):
    im = cv2.imread(str(path))
    h,w=im.shape[:2]
    bbox = anno['box']
    landmark = anno['landmark']
    landmark = quadrilateral_points2left_top_first_quadrilateral(landmark)
    new_bbox = expand_box(bbox,[1.1,1.4],w,h)
    #new_bbox = list(map(int,new_bbox))
    crop_landmark = landmark-new_bbox[:2]
    crop_img = im[int(new_bbox[1]):int(new_bbox[3]),int(new_bbox[0]):int(new_bbox[2])].copy()
    crop_bbox = bbox.reshape(-1,2)-new_bbox[:2].reshape(-1,2)
    h,w=crop_img.shape[:2]
    crop_h,crop_w = crop_bbox[1,1]-crop_bbox[0,1],crop_bbox[1,0]-crop_bbox[0,0]
    ih,iw=32,96

    if crop_h>ih and crop_w>iw:
        ratio = 1+(min(crop_h/ih,crop_w/iw)-1)/1.7
        resize_box = crop_bbox/[ratio,ratio]
        resize_landmark = crop_landmark/[ratio,ratio]
        resize_img = cv2.resize(crop_img,(int(w/ratio),int(h/ratio)))
       # cv2.rectangle(im,[int(resize_box[0,0]),int(resize_box[0,1])],[int(resize_box[1,0]),int(resize_box[1,1])],[255,0,0],2)

        return {
        'box': np.round(resize_box).astype(np.int16).reshape(-1).tolist(),
        'landmark': np.round(resize_landmark).astype(np.int16).reshape(-1).tolist(),
        'plate': anno['plate'],
        'img':cv2.imencode(".png", resize_img)[1]
        }

    return {
        'box': np.round(crop_bbox).astype(np.int16).reshape(-1).tolist(),
        'landmark': np.round(crop_landmark).astype(np.int16).reshape(-1).tolist(),
        'plate': anno['plate'],
        'img':cv2.imencode(".png", crop_img)[1]
    }

def process(path):
    annos = parse_crpd_label(path)
    out= []
    for anno in annos:
        res=process_single(path,anno)
        out.append(res)
    return out



if __name__ == '__main__':
    root = r"path/to/CRPD/"
    dst = r'dataset/CRPD/'
    Path(dst).mkdir(exist_ok=True,parents=True)
    write_record = mx.recordio.MXIndexedRecordIO(dst+'CRPD.idx',dst+'CRPD.rec','w')
    items = Path(root).rglob("*.jpg")
    i=0
    sum=0
    for item in tqdm.tqdm(items):
        try:
            anno=process(item)
        except:
            print(f'error:{str(item)}')
            continue
        for an in anno:
            item_byte = pkl.dumps(an)
            sum+=len(item_byte)
            write_record.write_idx(i,item_byte)
            i+=1
    print(f'{sum/1024/1024} mb')
    write_record.close()
    record = mx.recordio.MXIndexedRecordIO(dst+'CRPD.idx',dst+'CRPD.rec','r')
    item_byte = record.read_idx(1)
    item = pkl.loads(item_byte)
    print(item['plate'])