from pathlib import Path
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset
from skimage.transform import SimilarityTransform
import tqdm

from . import LicensePlateUtils


class CCPDDataset(Dataset):
    def __init__(self, root_dir: str, list_path: Optional[str] = None, align_plate: bool = False, preload=False, training=False):
        self.root_dir = Path(root_dir)
        if list_path is not None:
            list_path = Path(list_path)
        self.align_plate = align_plate
        self.preload = preload
        self.data: List[Tuple[Union[Image.Image, str, Path], Dict[str, Any]]] = []
        if list_path is not None:
            with list_path.open('r', encoding="utf-8") as f:
                for line in tqdm.tqdm(list(tqdm.tqdm(f, desc="Resolve dataset")), desc="Loading dataset"):
                    self.data.append(self._to_data_item(line.strip()))
        else:
            self.data = list(map(lambda x: self._to_data_item(x.relative_to(root_dir).as_posix()), tqdm.tqdm(self.root_dir.rglob("*.jpg"), desc="Loading dataset")))
        self.transforms = LicensePlateUtils.transforms(argument=training)

    def __len__(self):
        return len(self.data)

    def item(self, index):
        image, label = self.data[index]
        if isinstance(image, (str, Path)):
            image = self._read_image(self.root_dir / image, label)
        return image, label

    def __getitem__(self, index):
        image, label = self.item(index)
        tensor = self.transforms(image)
        return {
            "plate_image": image,
            "plate_image_tensor": tensor,
            "label": label['plate'],
            "label_tensor": LicensePlateUtils.encode(label['plate']),
        }

    @staticmethod
    def _parse_ccpd_label(image_path: Union[str, Path]):
        image_path = Path(image_path)
        provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
        alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
        ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
        area, title_degree, box, lamdmark, plate, brightness, blurriness = image_path.stem.split('-')
        plate_parts = plate.split('_')
        return {
            'area': float(f"0.{area}"),
            'title_degree': tuple(map(float, title_degree.split('_'))),
            'box': tuple(map(int, chain.from_iterable(map(lambda x: x.split('&'), box.split('_'))))),
            'lamdmark': np.fromiter((map(int, chain.from_iterable(map(lambda x: x.split('&'), lamdmark.split('_'))))), dtype=np.int32).reshape(-1, 2),
            'plate': f"{provinces[int(plate_parts[0])]}{alphabets[int(plate_parts[1])]}{''.join(map(lambda x: ads[int(x)], plate_parts[2:]))}",
            'brightness': float(brightness),
            'blurriness': float(blurriness),
        }

    def _read_image(self, image_path, label):
        image = Image.open(image_path)
        if self.align_plate:
            trans = SimilarityTransform()
            trans.estimate(np.float32(label['lamdmark']), np.float32(LicensePlateUtils.LANDMARK))
            M = trans.params[0:2, :]
            # M = cv2.getAffineTransform(np.float32(label['lamdmark'])[:3, :], np.float32(LicensePlateUtils.LANDMARK)[:3, :])
            aligned_plate_image = cv2.warpAffine(np.array(image), M, (130, 32), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            plate_image = Image.fromarray(aligned_plate_image)
        else:
            plate_image = image.crop(label['box'])
        return plate_image

    def _to_data_item(self, image_path) -> Tuple[Union[Image.Image, str, Path], Dict[str, Any]]:
        label = CCPDDataset._parse_ccpd_label(image_path)
        if self.preload:
            image = self._read_image(self.root_dir / image_path, label)
        else:
            image = image_path
        return (image, label)


if __name__ == '__main__':
    from torch.utils.data import ConcatDataset, DataLoader
    from lprec import utils
    CCPD2019_ROOT = Path(R"H:\CCPD2019")
    CCPD2020_ROOT = Path(R"H:\CCPD2020")
    dataset: ConcatDataset[CCPDDataset] = ConcatDataset([
        CCPDDataset(CCPD2019_ROOT, CCPD2019_ROOT / "splits/val.txt", preload=True),
        CCPDDataset(CCPD2020_ROOT / "ccpd_green/val", preload=True),
    ])
    dataloader = DataLoader(dataset, collate_fn=LicensePlateUtils.collate_fn, batch_size=8, shuffle=True, num_workers=0)
    for item in dataloader:
        print(item["plate_image_tensor"].shape)
        print(item["label_tensor"].shape)
        print(item["label_length"].shape)
        print(item["label"])
        utils.grid_plot_image(item["plate_image"], item["label"], 4)
        break
