import argparse
from pathlib import Path

import tqdm

from lprec.datasets import LicensePlateUtils, XZYLicensePlateDataset,MXRecordIODataset
from lprec.pl_model import PLModel
from lprec.pl_data import PLDataModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default='')
    parser.add_argument("--imgsz", nargs="+", type=int, default=[96,32],help="w,h")
    return parser.parse_args()


def test():
    args = parse_args()
    imgsz=args.imgsz
    
    model = PLModel.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.freeze()
    model.to("cuda")
    
    
    data = PLDataModule(test_datasets=[
        MXRecordIODataset(r'dataset/CBLPRD/CBLPRD.idx','dataset/CBLPRD/CBLPRD.rec',imgsz=imgsz),
        XZYLicensePlateDataset(Path("data/monitor/1"),imgsz=imgsz),
        XZYLicensePlateDataset(Path("data/monitor/2"),imgsz=imgsz),
    ], test_batch_size=1, test_num_workers=0)
    
    dataloader = data.test_dataloader()
    correct = 0
    total = 0
    progress = tqdm.tqdm(dataloader, total=len(dataloader))
    for batch in progress:
        batch_size = len(batch["label"])
        output = model(batch["plate_image_tensor"].to("cuda"))
        for y, label in zip(output.detach().cpu(), batch["label"]):
            try:
                predict = LicensePlateUtils.decode(y.squeeze())
            except:
                print(y, label )
                raise
            if predict == label:
                correct += 1
            total += 1
        progress.set_description(f"Batch: {batch_size}, Accuracy: {correct / total:.2%}, Total: {total}, Correct: {correct}, Wrong: {total - correct}")
    print(f"Accuracy: {correct / total}, Total: {total}, Correct: {correct}, Wrong: {total - correct}")


if __name__ == "__main__":
    test()
