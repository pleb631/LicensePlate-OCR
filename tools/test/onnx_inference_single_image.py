import argparse
import cv2
import numpy as np
import onnxruntime
from PIL import Image

from lprec.datasets import LicensePlateUtils
from lprec import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="data/test.png")
    parser.add_argument("--onnx", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    model = onnxruntime.InferenceSession(str(args.onnx), providers=["CUDAExecutionProvider"])
    input_name = model.get_inputs()[0].name

    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    landmark = np.array([
        [614, 575],
        [429, 577],
        [430, 526],
        [615, 523]
    ], dtype=np.float32)

    M = cv2.getAffineTransform(landmark[:3, :], np.float32(LicensePlateUtils.LANDMARK)[:3, :])
    aligned_plate_image = cv2.warpAffine(image, M, (130, 32), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    plate_image = Image.fromarray(aligned_plate_image)

    x = LicensePlateUtils.transforms()(plate_image).unsqueeze(0)
    output = model.run(None, {input_name: x.numpy()})[0]
    predict_plates = []
    for item in output:
        predict_plates.append(LicensePlateUtils.decode(item.squeeze()))
    print(predict_plates)
    utils.grid_plot_image([plate_image], predict_plates, 4)


if __name__ == "__main__":
    main()
