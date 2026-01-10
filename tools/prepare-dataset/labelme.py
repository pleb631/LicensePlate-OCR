from pathlib import Path
import pickle as pkl
import cv2
import tqdm
import mxnet as mx
import json
import os
import argparse


def containExtraAlphabet(line):
    if line[0] not in "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新":
        return True
    for alphabet in line[1:]:
        if alphabet not in "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ":
            return True
    return False

def read_json(anno_path):
    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data 
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default='')
    parser.add_argument("--dst", type=str, default='')
    parser.add_argument("--name", type=str, default='lp-labelme')
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root
    dst  =  args.dst 
    name = args.name
    
    # os.mkdir(dst,)
    items = Path(root).rglob("*.png")
    txt_data = []
    i = 0
    sum = 0
    write_record = mx.recordio.MXIndexedRecordIO(
        os.path.join(dst,f"{name}.idx"), os.path.join(dst,f"{name}.rec"), "w"
    )
    for image_path in tqdm.tqdm(items):
        json_path = image_path.with_suffix(".json")
        info = read_json(json_path)
        points = info["shapes"][0]["points"]
        plate = info["shapes"][0]["label"]
        if containExtraAlphabet(plate):
            continue
        im = cv2.imread(str(image_path))
        h, w, _ = im.shape
        points = [int(round(i)) for p in points for i in p]
        bbox = [points[0], points[1], points[2], points[3]]

        anno = {
            "box": bbox,
            "plate": plate,
            "img": cv2.imencode(".png", im)[1],
        }
        item_byte = pkl.dumps(anno)
        sum += len(item_byte)
        write_record.write_idx(i, item_byte)
        i += 1

    print(f"{sum/1024/1024} mb")
    write_record.close()
    record = mx.recordio.MXIndexedRecordIO(os.path.join(dst,f"{name}.idx"), os.path.join(dst,f"{name}.rec"), "r")
    item_byte = record.read_idx(1)
    item = pkl.loads(item_byte)
    print(item["plate"])

if __name__ == "__main__":
    main()