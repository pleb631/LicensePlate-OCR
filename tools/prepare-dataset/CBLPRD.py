import pickle as pkl
import cv2
import tqdm
import mxnet as mx
import os


def containExtraAlphabet(line):
    if line[0] not in "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新":
        return True
    for alphabet in line[1:]:
        if alphabet not in "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ":
            return True
    return False


if __name__ == "__main__":
    root = r"path/to/CBLPRD/"

    dst = r"dataset/CBLPRD/"
    txt_file = open(root + "data.txt", "r", encoding="utf-8-sig")
    txt_data = []
    i = 0
    sum = 0
    write_record = mx.recordio.MXIndexedRecordIO(
        dst + "CBLPRD.idx", dst + "CBLPRD.rec", "w"
    )
    for line in tqdm.tqdm(txt_file.readlines()):
        line = line.strip().split()
        if "双层" in line[-1]:
            # print(f"pass {line[-1]}")
            continue
        if containExtraAlphabet(line[-2]):
            continue
        image_path = os.path.join(root, line[0])
        im = cv2.imread(image_path)
        h, w, _ = im.shape
        bbox = [0, 0, im.shape[1], im.shape[0]]
        landmark = [0, 0, w, 0, w, h, 0, h]
        plate = line[1]
        anno = {
            "box": bbox,
            "plate": plate,
            "img": cv2.imencode(".png", im)[1],
            "landmark": landmark,
        }
        item_byte = pkl.dumps(anno)
        sum += len(item_byte)
        write_record.write_idx(i, item_byte)
        i += 1

    print(f"{sum/1024/1024} mb")
    write_record.close()
    record = mx.recordio.MXIndexedRecordIO(dst + "CBLPRD.idx", dst + "CBLPRD.rec", "r")
    item_byte = record.read_idx(1)
    item = pkl.loads(item_byte)
    print(item["plate"])
