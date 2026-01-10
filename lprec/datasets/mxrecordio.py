import pickle
from pathlib import Path
import torch
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
import cv2
import mxnet
import tqdm
import numpy as np
import albumentations as A
if __name__=='__main__':
    from license_plate_utils import LicensePlateUtils

else:
    from . import LicensePlateUtils
    
from lprec.utils import xyxy2xywh_center,xywh_center2xyxy


def shift_bbox(xyxy,h,w):
    xywh = xyxy2xywh_center(xyxy)
    xywh[0] = xywh[0]+ np.random.randint(-5,5)
    xywh[1] = xywh[1]+ np.random.randint(-5,5)
    bbox = xywh_center2xyxy(xywh)
    bbox[0]=np.clip(bbox[0],0,w)
    bbox[2]=np.clip(bbox[2],0,w)
    bbox[1]=np.clip(bbox[1],0,h)
    bbox[3]=np.clip(bbox[3],0,h)
    return bbox
    

class ImageArgument:
    def __init__(self, training=False, with_keypoint=False,ShiftScaleRotate=False,imgsz=(96, 32),*args, **kwargs) -> None:
        super().__init__()

        self.image_size = imgsz
        self.mean = (0, 0, 0)
        self.std = (1,1,1)
        self.training = training
         
        self.imgOnlyAugment = A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(),
                        A.ColorJitter(),
                    ]
                ),
                A.RandomGamma(),
                A.OneOf(
                    [
                        A.Blur(blur_limit=5),
                        A.MotionBlur(blur_limit=5),
                        A.MedianBlur(blur_limit=5),
                    ]
                ),
                A.ToGray(p=0.01),
                A.RGBShift(),
                A.GaussNoise(),
            ]
        )
        self.with_keypoint=with_keypoint
        if self.with_keypoint:
            Augment=[]
            if ShiftScaleRotate:
                Augment.append(A.ShiftScaleRotate(rotate_limit=10,p=0.5,shift_limit=0,scale_limit=0,border_mode=cv2.BORDER_CONSTANT))
                
            self.Augment = A.Compose(
                Augment,
                keypoint_params=A.KeypointParams(format="xy",remove_invisible=False),
            )

    def __call__(self, image, bbox, landmark):
        h,w,c=image.shape
        r = min(self.image_size[0] / w, self.image_size[1] / h)
        if self.training:
            interp = cv2.INTER_LINEAR
        else:
            r = min(self.image_size[0] / w, self.image_size[1] / h)
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
        if self.training:
            image = self.imgOnlyAugment(image=image)["image"]
        
            if self.with_keypoint:
                landmark1 = np.array(landmark).reshape(-1,2)
                items = self.Augment(image=image,keypoints=landmark1)
                image,landmark = items['image'],items['keypoints']
                
                bbox = np.array([np.min(landmark,axis=0),np.max(landmark,axis=0)])
                bbox[:,0]=np.clip(bbox[:,0],0,w)
                bbox[:,1]=np.clip(bbox[:,1],0,h)
                bbox = np.round(bbox.reshape(-1)).astype(np.int16)
            bbox = shift_bbox(bbox,h,w)
        image = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        image = cv2.resize(image, self.image_size,interpolation=interp)
        image = torch.from_numpy(image.astype("float32")).permute(2, 0, 1).div_(255.0)
        image = F.normalize(image, mean=self.mean, std=self.std, inplace=True)
        return image


class MXRecordIODataset(Dataset):
    def __init__(self, idx_path, rec_path, training=False, *args, **kwargs) -> None:
        super().__init__()
        self.record = mxnet.recordio.MXIndexedRecordIO(
            str(idx_path), str(rec_path), "r"
        )
        self.transforms = ImageArgument(training,*args, **kwargs)
        print(f"Load {idx_path}, length={len(self.record.idx)}")

    def __len__(self) -> int:
        return len(self.record.idx)

    def __getitem__(self, index: int):
        item = pickle.loads(self.record.read_idx(index))
        image = cv2.imdecode(item["img"], 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = item["plate"]
        bbox = item["box"]
        landmark = item.get("landmark",[])
        # cv2.rectangle(image, [bbox[0], bbox[1]], [bbox[2], bbox[3]], [255, 0, 0], 2)
        # for point in np.array(landmark).reshape(-1, 2):
        #     cv2.circle(image, point, 5, [255, 0, 0], -1)
        # cv2.imwrite("test.jpg", image)
        tensor = self.transforms(image, bbox, landmark)
        target = LicensePlateUtils.encode(label)

        return {
            "plate_image": image,
            "plate_image_tensor": tensor,
            "label": label,
            "label_tensor": target,
        }


def grid_plot_image(images, labels, cols=4, **kwargs):
    import matplotlib.pyplot as plt

    n = len(images)
    rows = (n + cols - 1) // cols
    if kwargs:
        plt.figure(**kwargs)
    for index in range(n):
        plt.subplot(rows, cols, index + 1)
        plt.title(str(labels[index]))
        plt.axis("off")
        plt.imshow(images[index])
    plt.savefig("fig.png")


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # import utils
    dataset = MXRecordIODataset(
        Path("dataset/CCPD/CCPD.idx"),
        Path("dataset/CCPD/CCPD.rec"),
        with_keypoint=True,
        ShiftScaleRotate=True,
    )
    dataloader = DataLoader(
        dataset,
        collate_fn=LicensePlateUtils.collate_fn,
        batch_size=8,
        shuffle=False,
        num_workers=1,
    )
    for item in tqdm.tqdm(dataloader):
        # print(item["plate_image_tensor"].shape)
        # print(item["label_tensor"].shape)
        # print(item["label_length"].shape)
        # print(item["label"])
        grid_plot_image(item["plate_image"], item["label"], 4)
        break
        ...
