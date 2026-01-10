import argparse
from pathlib import Path

import tqdm
import onnxruntime

from lprec.datasets import LicensePlateUtils, XZYLicensePlateDataset,MXRecordIODataset
from lprec.pl_data import PLDataModule
from lprec.utils import save_txt



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default='')
    return parser.parse_args()



def test():
    args = parse_args()
    model = onnxruntime.InferenceSession(args.onnx, providers=["CUDAExecutionProvider"])
    input_name = model.get_inputs()[0].name
    imgsz = model.get_inputs()[0].shape[3:1:-1]
    
    data = PLDataModule(test_datasets=[
        MXRecordIODataset(r'dataset/CBLPRD/CBLPRD.idx','dataset/CBLPRD/CBLPRD.rec',imgsz=imgsz),
        XZYLicensePlateDataset(Path("data/monitor/1"),imgsz=imgsz),
        XZYLicensePlateDataset(Path("data/monitor/2"),imgsz=imgsz),
    ], test_batch_size=1, test_num_workers=0)
    
    dataloader = data.test_dataloader()
    correct = 0
    total = 0
    progress = tqdm.tqdm(dataloader, total=len(dataloader))
    badcase=[]
    for batch in progress:
        batch_size = len(batch["label"])
        output = model.run(None, {input_name: batch["plate_image_tensor"].cpu().numpy()})[0]
        for y, label in zip(output, batch["label"]):
            
            predict = LicensePlateUtils.decode(y.squeeze().transpose((1,0)))

            if predict == label:
                correct += 1
            else:
                
                img_path = batch.get("img_path",[])
                if len(img_path)>0:
                    badcase.append(img_path[0])
                
            total += 1
        progress.set_description(f"Batch: {batch_size}, Accuracy: {correct / total:.2%}, Total: {total}, Correct: {correct}, Wrong: {total - correct}")
    print(f"Accuracy: {correct / total}, Total: {total}, Correct: {correct}, Wrong: {total - correct}")
    save_txt('basecase.txt',badcase)

if __name__ == "__main__":
    test()
