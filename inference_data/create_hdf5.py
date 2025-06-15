import os
#import io
import h5py
import json
import argparse
import numpy as np
import csv
#from PIL import Image
#from tqdm import tqdm
import ast

UNDERSCORE_SORT = lambda s: list(map(try_int, s.split("_")))

def load_csv_to_dict(csvpath):
    """Load CSV file into a dictionary, allowing multiple values for the same key."""
    lookup_dict = {}
    with open(csvpath, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = (int(row['page_no']), int(row['panel_no']), int(row['comic_no']), int(row['dialog_or_narration']))
            text = row['text']
            if key not in lookup_dict:
                lookup_dict[key] = row['text'] 
            else:
                lookup_dict[key] += (' ' + text)
    return lookup_dict

def load_csv_to_dict2(csvpath):
    
    
    with open(csvpath, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    # Create a new dictionary with the desired keys
    new_data = {}
    
    for key, value in data.items():
        # Convert the string key to a tuple of integers
        new_key = ast.literal_eval(key)  # safely converts string to tuple
        
        # Ensure the key is a tuple of integers (optional, if the format is guaranteed)
        new_key = tuple(map(int, new_key))
        
        # Change the order of the tuple (3, 1, 2 instead of 1, 2, 3)
        new_key = (new_key[1], new_key[2], new_key[0])
        
        # Add the modified key-value pair to the new dictionary
        new_data[new_key] = value

    return new_data


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
            "str_data1",
            (len(img_list),),
            dtype=dt,
        )
        grp.create_dataset(
            "str_data2",
            (len(img_list),),
            dtype=dt,
        )
        for i, img_num in enumerate(img_idx):
            
            split_string = img_num.split('_')
            key = (int(split_string[0]), int(split_string[1]), int(number_storage), 1)
            grp["str_data1"][i] = lookupdict.get(key, '')
            
            key2 = (int(split_string[0]), int(split_string[1]), int(number_storage), 2)
            grp["str_data2"][i] = lookupdict.get(key2, '')
         
    return path_index


def main(args):

    path = args.data_path
    csv_path = args.csv_path
    dataset_name = path.split("/")[-1]
    writer = h5py.File(f"{path}_{args.modality}.hdf5", "w")
    
    lookup_dict = load_csv_to_dict(csv_path)
    


    index = {}
    index[dataset_name] = process_dir(path, dataset_name, writer, lookup_dict)

    writer.close()

    with open(f"{path}_{args.modality}_indexing.json", "w+") as f:
        json.dump(index, f)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories to hdf5")
    parser.add_argument("--data-path", type=str, default="Data/comics/raw_panel_images/raw_panel_images/train_images")
    parser.add_argument("--csv-path", type=str, default="Data/comics/simplified_COMICS_TEXT_PLUS_ocr_file.csv")
    parser.add_argument("--modality", type=str, default="text")
    args = parser.parse_args()
    main(args)