
import pickle
import pickledb
import tqdm
import os
import pprint

def main():


    db_path = 'data.db'
    db = pickledb.load(db_path,auto_dump=False)
    keys = db.getall()
    result ={}
    for key in tqdm.tqdm(keys):
        if key[0] not in result:
            result[key[0]]=0
        result[key[0]]+=db.get(key)
    
    result = result.items()
    result = sorted(result,key=lambda x: x[1],reverse=True)
    pprint.pprint(result)


main()