import os
import argparse
#import pandas as pd

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="Data/comics/raw_panel_images/raw_panel_images")
    parser.add_argument("--level", type=str, choices=["panel", "character"], default="panel")
    args = parser.parse_args()

    comics = os.listdir(f"{args.data_dir}")

    train = comics[: int(0.8 * len(comics))]
    val = comics[int(0.8 * len(comics)) : int(0.9 * len(comics))]
    test = comics[int(0.9 * len(comics)) :]
  
    splits = {"train": train, "val": val, "test": test}

    for split in ["train", "val", "test"]:
        
        if not os.path.exists(f"{args.data_dir}/{split}_images"):
            
            os.mkdir(f"{args.data_dir}/{split}_images")
            
        for comic in splits[split]:
            
            if os.path.exists(f"{args.data_dir}/{comic}"):
                
                source_dir = os.path.join(args.data_dir, comic)
                destination_dir = os.path.join(args.data_dir, f"{split}_images", comic)
                
                if not os.path.exists(os.path.dirname(destination_dir)):
                    
                    os.makedirs(os.path.dirname(destination_dir))

                # Windows version with xcopy for recursive copying
                os.system(f'xcopy /E /I "{source_dir}" "{destination_dir}"')