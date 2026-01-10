from pathlib import Path
import json
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
from torchvision.transforms import functional as F


from . import LicensePlateUtils
from lprec.utils import imread



class XZYLicensePlateDataset(Dataset):
    def __init__(self, image_dir: Path, label_dir: Path = None, training=False,imgsz = (96, 32)) -> None:
        super().__init__()
        if label_dir is None:
            label_dir = image_dir / "annotations"
            image_dir = image_dir / "images"
        self.data = []
        self.imgsz = imgsz
        for anno_file in label_dir.rglob("*.json"):
            anno = json.loads(anno_file.read_text(encoding='utf-8'))
            plates = anno['annotation']['license']
            img_paths = list((image_dir / anno_file.parent.relative_to(label_dir)).glob(f"{anno_file.stem}.*"))
            if len(img_paths) != 1:
                # import os 
                # os.remove(str(anno_file))
                continue
            img_path = img_paths[0]
            for plate in plates:
                label = plate['text']
                box = plate['box']
                self.data.append((img_path, label, box))
        self.transforms = LicensePlateUtils.transforms(argument=training)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        img_path, label, bbox = self.data[index]

        origin_image = imread(img_path)
        h,w,c=origin_image.shape
        bbox=[bbox[0]*w,bbox[1]*h,bbox[2]*w,bbox[3]*h]
        
        bbox = list(map(int,bbox))
        image = origin_image[bbox[1]:bbox[3],bbox[0]:bbox[2],:].copy()
        image = image[:,:,::-1]
        image = cv2.resize(image, dsize=self.imgsz)
        tensor = torch.from_numpy(image.astype("float32")).permute(2, 0, 1).div_(255.0)
        tensor = F.normalize(tensor, mean=(0,0,0), std=(1,1,1), inplace=True)
        target = LicensePlateUtils.encode(label)
        img_path = str(img_path)
        
        return {
            "img_path":img_path,
            "plate_image": image,
            "plate_image_tensor": tensor,
            "label": label,
            "label_tensor": target,
        }


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from lprec import utils
    dataset = XZYLicensePlateDataset(Path("dataset/monitor/images"), Path("dataset/monitor/annotations"))
    dataloader = DataLoader(dataset, collate_fn=LicensePlateUtils.collate_fn, batch_size=8, shuffle=True)
    for item in dataloader:
        print(item["plate_image_tensor"].shape)
        print(item["label_tensor"].shape)
        print(item["label_length"].shape)
        print(item["label"])
        utils.grid_plot_image(item["plate_image"], item["label"], 4)
        break
