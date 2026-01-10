from pathlib import Path
import argparse
import os

import onnx
from onnxsim import simplify
import torch
import torch.nn as nn

from lprec.pl_model import PLModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default='')
    return parser.parse_args()

class ModelWrap(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        return y


def main():
    args = parse_args()
    model = PLModel.load_from_checkpoint(args.checkpoint)
    model.model.export=True
    model.eval()
    model.freeze()
    model.to("cpu")

    output = str(Path(args.checkpoint).with_suffix(".onnx"))
    print("export to", output)
    torch.onnx.export(
        ModelWrap(model), torch.rand(1,3,32,96), output,
        input_names=["input"], output_names=["output"],
        keep_initializers_as_inputs=False, verbose=False, opset_version=11,
        # dynamic_axes={"input": {0: "batch"}}
    )
    model = onnx.load(output)
    os.remove(output)
    print("simplifying")
    model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)
    


if __name__ == "__main__":
    main()
