import argparse

from lprec.pl_model import PLModel
from lprec.pl_data import PLDataModule
from datasets import LicensePlateUtils, XZYLicensePlateDataset
from lprec import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    model = PLModel.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.freeze()
    model.to("cuda")

    data = PLDataModule(test_datasets=[
        XZYLicensePlateDataset("data/monitor/1"),
        XZYLicensePlateDataset("data/monitor/2"),
    ], test_batch_size=1, test_num_workers=0)
    dataloader = data.test_dataloader()
    for i, batch in enumerate(dataloader):
        output = model(batch["plate_image_tensor"].to("cuda"))
        predict_plates = []
        for item in output.detach():
            predict_plates.append(LicensePlateUtils.decode(item.cpu().numpy()))
        utils.grid_plot_image(batch["plate_image"], list(map(lambda x: f"{x[0] == x[1]}\nPredict='{x[0]}'\nLabel  ='{x[1]}'", zip(predict_plates, batch["label"]))), 4)
        if i >= 4:
            break


if __name__ == "__main__":
    main()
