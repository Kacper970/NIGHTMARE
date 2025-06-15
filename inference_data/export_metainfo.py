import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="Data/comics/raw_panel_images/raw_panel_images")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-bins", type=int, default=100)
    args = parser.parse_args()

    split = f"{args.split}_images"

    metainfo = {}
    comics = os.listdir(os.path.join(args.data_dir, split))

    for comic in tqdm(comics):
        images = os.listdir(os.path.join(args.data_dir, split, comic))

        for image in images:
            img = Image.open(os.path.join(args.data_dir, split, comic, image))

            width, height = img.size
            ratio = width / height

            img_hsv = img.convert("HSV")
            hue = np.mean(np.array(img_hsv)[:, :, 0])
            saturation = np.mean(np.array(img_hsv)[:, :, 1])
            intensity = np.mean(np.array(img_hsv)[:, :, 2])
            color = saturation > 0

            metainfo[f"{comic}/{image.replace('.jpg', '')}"] = {
                "width": width,
                "height": height,
                "ratio": ratio,
                "color": color,
                "intensity": intensity,
                "saturation": saturation,
                "hue": hue,
                "channel_std": None,
            }

    # Create df where each row is an image
    metainfo = pd.DataFrame.from_dict(
        {i: metainfo[i] for i in metainfo.keys()}, orient="index"
    )

    metainfo["ratio_bin"] = pd.qcut(
        metainfo["ratio"], args.num_bins, labels=False, duplicates="drop"
    )

    metainfo["intensity_bin"] = pd.qcut(
        metainfo["intensity"], args.num_bins, labels=False, duplicates="drop"
    )
    metainfo["saturation_bin"] = pd.qcut(
        metainfo["saturation"], args.num_bins, labels=False, duplicates="drop"
    )

    metainfo.to_csv(f"{args.data_dir}/metainfo_{args.split}.csv")

    # Plot ratio distribution
    plt.hist(metainfo["ratio"], bins=100)
    plt.xlabel("Ratio")
    plt.ylabel("Count")
    plt.grid(True)
    plt.title("Ratio Distribution")
    plt.savefig(f"{args.data_dir}/{args.split}_ratio_distribution.png")
    plt.clf()

    # Plot intensity distribution
    plt.hist(metainfo["intensity"], bins=100)
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.grid(True)
    plt.title("Intensity Distribution")
    plt.savefig(f"{args.data_dir}/{args.split}_intensity_distribution.png")
    plt.clf()

    # Plot saturation distribution
    plt.hist(metainfo["saturation"], bins=100)
    plt.xlabel("Saturation")
    plt.ylabel("Count")
    plt.grid(True)
    plt.title("Saturation Distribution")
    plt.savefig(f"{args.data_dir}/{args.split}_saturation_distribution.png")
    
    plt.clf()
