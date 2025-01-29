import os
#import io
import h5py
import json
import argparse
import numpy as np
import csv
#from PIL import Image
#from tqdm import tqdm

UNDERSCORE_SORT = lambda s: list(map(try_int, s.split("_")))

def load_csv_to_dict(csvpath):
    """Load CSV file into a dictionary, allowing multiple values for the same key."""
    lookup_dict = {}
    with open(csvpath, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = (int(row['page_no']), int(row['panel_no']), int(row['comic_no']))
            text = row['text']
            if key not in lookup_dict:
                lookup_dict[key] = row['text'] 
            else:
                lookup_dict[key] += (' ' + text)
    return lookup_dict

def try_int(s):
    try:
        return int(s)
    except ValueError:
        return int.from_bytes(s.encode(), "little")

def process_dir(path, name, writer, lookupdict, number_storage=None):
    
    contents = os.listdir(path)
    grp = writer.create_group(name)
    path_index = {}
    img_list, img_idx = [], []
   

    for content in sorted(contents, key=UNDERSCORE_SORT):
        content_l = content.lower()
        if os.path.isdir(f"{path}/{content}"):
            path_index[content] = process_dir(f"{path}/{content}", content, grp, lookupdict, number_storage=content)
        elif (
            content_l.endswith(".jpg")
            or content_l.endswith(".jpeg")
            or content_l.endswith(".png")
        ):
            entry_name = content.split(".")[0]
            with open(f"{path}/{content}", "rb") as f:
                img_data = f.read()
            

            img_list.append(img_data)
            img_idx.append(entry_name)

    if len(img_list) > 0:
        path_index["img_data"] = img_idx
        dt = h5py.special_dtype(vlen=np.dtype("uint8"))
        grp.create_dataset(
            "img_data",
            (len(img_list),),
            dtype=dt,
        )
        for i, img in enumerate(img_list):
            grp["img_data"][i] = np.frombuffer(img, dtype="uint8")
                   
    if (len(img_list) > 0):
        path_index["str_data"] = img_idx
        dt = h5py.special_dtype(vlen=str)
        grp.create_dataset(
            "str_data",
            (len(img_list),),
            dtype=dt,
        )
        for i, img_num in enumerate(img_idx):
            
            split_string = img_num.split('_')
            key = (int(split_string[0]), int(split_string[1]), int(number_storage))
            grp["str_data"][i] = lookupdict.get(key, '')
         
    return path_index


def main(args):

    path = args.data_path
    csv_path = args.csv_path
    dataset_name = path.split("/")[-1]
    writer = h5py.File(f"{path}.hdf5", "w")
    
    lookup_dict = load_csv_to_dict(csv_path)

    index = {}
    index[dataset_name] = process_dir(path, dataset_name, writer, lookup_dict)

    writer.close()

    with open(f"{path}_indexing.json", "w+") as f:
        json.dump(index, f)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories to hdf5")
    parser.add_argument("--data-path", type=str, default="Data/comics/raw_panel_images/raw_panel_images/code_test")
    parser.add_argument("--csv-path", type=str, default="Data/comics/COMICS_ocr_file.csv")
    args = parser.parse_args()
    main(args)