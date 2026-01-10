from pathlib import Path
import argparse

import onnxruntime

from lprec.pl_data import PLDataModule
from lprec.datasets import LicensePlateUtils, XZYLicensePlateDataset
from lprec import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default='')
    return parser.parse_args()


def get_title(x):
    if x[0] == x[1]:
        return f"True\n{x[0]}"
    else:
        return f"False\npredict='{x[0]}'\nlabel='{x[1]}'"


def main():
    args = parse_args()
    model = onnxruntime.InferenceSession(str(args.onnx), providers=["CUDAExecutionProvider"])
    input_name = model.get_inputs()[0].name

    data = PLDataModule(test_datasets=[
        XZYLicensePlateDataset(Path("data/monitor/1")),
        XZYLicensePlateDataset(Path("data/monitor/2")),
    ], test_batch_size=12, test_num_workers=0)
    for i, batch in enumerate(data.test_dataloader()):
        output = model.run(None, {input_name: batch["plate_image_tensor"].numpy()})[0]
        predict_plates = []
        for item in output:
            predict_plates.append(LicensePlateUtils.decode(item.squeeze()))
        utils.grid_plot_image(batch["plate_image"], list(map(get_title, zip(predict_plates, batch["label"]))), 4)
        if i >= 4:
            break


if __name__ == "__main__":
    main()
