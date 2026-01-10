from PIL import Image
from torch.utils.data import Dataset
import cv2

from . import LicensePlateUtils
from .license_plate_generator import LicensePlateGenerator


class FakeLicensePlateDataset(Dataset):
    def __init__(self, dataset_size=100_000, training=False) -> None:
        super().__init__()
        self.dataset_size = dataset_size
        self.transforms = LicensePlateUtils.transforms(argument=training)
        self.generator = LicensePlateGenerator()

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, _: int):
        image, label = self.generator.generate_license_plate_images()
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = self.transforms(image)
        target = LicensePlateUtils.encode(label)
        return {
            "plate_image": image,
            "plate_image_tensor": tensor,
            "label": label,
            "label_tensor": target,
        }


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from lprec import utils
    dataset = FakeLicensePlateDataset(training=True)
    dataloader = DataLoader(dataset, collate_fn=LicensePlateUtils.collate_fn, batch_size=16, num_workers=0, shuffle=True)
    for i, item in enumerate(dataloader):
        print(item["plate_image_tensor"].shape)
        print(item["label_tensor"].shape)
        print(item["label_length"].shape)
        print(item["label"])
        utils.grid_plot_image(item["plate_image"], item["label"], 4)
        if i >= 4:
            break
