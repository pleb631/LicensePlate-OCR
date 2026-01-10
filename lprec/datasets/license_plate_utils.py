import torch
import numpy as np
from PIL import Image
import cv2


class ImageArgument(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        import albumentations as A
        self.argument = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.Blur(p=0.5),
            A.MotionBlur(p=0.5),
            A.GaussNoise(p=0.5),
        ])

    def forward(self, image: Image.Image):
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image = self.argument(image=image)["image"]
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image


class LicensePlateUtils(object):
    _chars: list[str] = [
        '-',
        '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉',
        '黑', '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂',
        '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕',
        '甘', '青', '宁', '新',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
        'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
        'W', 'X', 'Y', 'Z'
    ]
    _chars_dict = {char: i for i, char in enumerate(_chars)}

    LANDMARK = np.array([
        [125, 27.0],
        [5.0, 27.0],
        [5.0, 5.0],
        [125, 5.0],
    ])

    @staticmethod
    def encode(label_str):
        import torch
        return torch.tensor([LicensePlateUtils._chars_dict[char] for char in label_str], dtype=torch.long)

    @staticmethod
    def decode(label_tensor):
        import numpy as np
        blank = LicensePlateUtils.blank()
        chars = []
        prechar = None
        # print(label_tensor.shape)
        if len(label_tensor.shape) == 2:
            label_tensor = np.argmax(label_tensor, axis=0)
        for i in label_tensor:
            if i != blank and i != prechar:
                try:
                    chars.append(LicensePlateUtils._chars[i])
                except:
                    
                    raise
            prechar = i
        return ''.join(chars)

    @staticmethod
    def blank():
        return LicensePlateUtils._chars_dict['-']

    @staticmethod
    def num_classes():
        return len(LicensePlateUtils._chars)

    @staticmethod
    def transforms(image_size=(96, 32), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), argument=False):
        import torchvision.transforms as T
        arguments = []
        if argument:
            arguments.append(ImageArgument())
        return T.Compose([
            *arguments,
            T.Resize(image_size[::-1]),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    @staticmethod
    def collate_fn(batch):
        import torch
        plate_images = []
        plate_image_tensors = []
        labels = []
        label_tensors = []
        lengths = []
        img_path=[]
        
        for item in batch:
            if "img_path" in item:
                img_path.append(item["img_path"])
            plate_images.append(item['plate_image'])
            plate_image_tensors.append(item['plate_image_tensor'])
            labels.append(item['label'])
            label_tensors.append(item['label_tensor'])
            lengths.append(len(item['label']))
        return {
            "img_path":img_path,
            "plate_image": plate_images,
            "plate_image_tensor": torch.stack(plate_image_tensors),
            "label": labels,
            "label_tensor": torch.concat(label_tensors, dim=0),
            "label_length": torch.tensor(lengths)
        }
