from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import tqdm

from . import LicensePlateUtils


class LicensePlateDataset(Dataset):
    def __init__(self, root_dir: Path, label_path: Path, preload=False, training=False) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.label_path = label_path
        self.preload = preload
        self.data = []
        with label_path.open('r', encoding="utf-8") as f:
            for line in tqdm.tqdm(list(tqdm.tqdm(f, desc="Resolve dataset")), desc="Loading dataset"):
                image_path, label = line.strip().split()
                if self.preload:
                    image = Image.open(self.root_dir / image_path)
                    self.data.append((image, label))
                else:
                    self.data.append((image_path, label))
        self.transforms = LicensePlateUtils.transforms(argument=training)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        image, label = self.data[index]
        if isinstance(image, (str, Path)):
            image = Image.open(self.root_dir / image)
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
    dataset = LicensePlateDataset(Path("G:\\train"), Path("G:\\\\train.txt"))
    dataloader = DataLoader(dataset, collate_fn=LicensePlateUtils.collate_fn, batch_size=8, shuffle=True)
    for item in dataloader:
        print(item["plate_image_tensor"].shape)
        print(item["label_tensor"].shape)
        print(item["label_length"].shape)
        print(item["label"])
        utils.grid_plot_image(item["plate_image"], item["label"], 4)
        break
