import mxnet
import pickle
import pickledb
from tqdm import trange



class MXRecordIODataset:
    def __init__(self, idx_path, rec_path) -> None:
        super().__init__()
        self.record = mxnet.recordio.MXIndexedRecordIO(
            str(idx_path), str(rec_path), "r"
        )

        print(f"Load {idx_path}, length={len(self.record.idx)}")

    def __len__(self) -> int:
        return len(self.record.idx)

    def __getitem__(self, index: int):
        item = pickle.loads(self.record.read_idx(index))
        label = item["plate"]
        
        return label


def main(idx_path):
    import os
    # idx_path = 'dataset/CCPD/CCPD.idx'
    rec_path = idx_path[:-4]+'.rec'
    dataset_name = os.path.basename(idx_path).split('.')[0] 
    db_path = 'data.db'
    db = pickledb.load(db_path,auto_dump=False)
    if db.exists("dataset_names"):
        dataset_names = db.get("dataset_names")
        assert dataset_name not in dataset_names
    data = MXRecordIODataset(idx_path,rec_path)
    for i in trange(len(data)):
        plate = data[i]
        if db.exists(plate):
            db.set(plate,db.get(plate)+1)
        else:
            db.set(plate,db.get(plate)+1)
    db.dump()
    
    if db.exists("dataset_names"):
        dataset_names = db.get("dataset_names")
        db.set("dataset_names",dataset_names+dataset_name)
    else:
        db.set("dataset_names",[dataset_name])
    
def batch_predict():
    root =[
        r'dataset/CCPD/CCPD.idx',
           "dataset/CRPD/CRPD.idx",
           "dataset/CBLPRD/CBLPRD.idx",
        ]
    for i in root:
        main(i)
    
if __name__=='__main__':
    batch_predict()
    # main()