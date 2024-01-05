from os.path import join

import config
import labelmappings as lm
import pandas as pd
from monai.transforms import LoadImage, MapLabelValue, SaveImage
from tqdm import tqdm

# Specify channel mappings
channels_all = set(range(1, 41))
channels_charite = {1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 20, 21, 22, 23, 31, 32, 33, 34}
channels_MRSeg_V1 = {4, 31, 32, 33, 34}
channels_MRSeg_V2 = {4, 10, 11, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40}

# Clean segmentations
Clean = lambda channels: MapLabelValue(
    orig_labels=channels,
    target_labels=[0] * len(channels),
    dtype=int,
)

# Save result
Save = lambda path: SaveImage(
    output_dir=join(config.ukbb, path),
    separate_folder=False,
    output_postfix="new",
    print_log=False,
)


def merge_label(bottom_label, top_label, top_label_channels):
    """Merge two labels. Channels spefied in ´top_label_channels´ will be deleted from the bottom label and vice versa.
    If channels from both labels overlap, the channel of the top label will be kept.
    Informal: This function puts the top label "on top" of the bottom label"""

    # clean labels
    bottom_label = Clean(top_label_channels)(bottom_label)
    top_label = Clean(channels_all - top_label_channels)(top_label)

    # merge labels
    new_label = bottom_label.detach().clone()
    new_label[top_label != 0] = top_label[top_label != 0]

    return new_label


def merge_sections_123(segmentation_generation):
    """Merge predictions of MRSegmentator and TotalSegmentator of sections one, two and three."""
    data = pd.read_csv(join(config.ukbb, "manifest.csv"))

    # remove patients with more than 6 scans? Might be erroneous?
    patients2remove = data.loc[data["section"] == 6]["eid"].unique()
    _index = data.loc[data["eid"].apply(lambda x: x in patients2remove)].index
    data = data.drop(_index).reset_index(drop=True)
    print(f"Dropped {len(patients2remove)} patients")

    # Select water only images of section 1,2,3
    data = data.loc[data["dixon_type"] == "W"]
    data = data.loc[data["section"].apply(lambda x: x in [1, 2, 3])]
    data = data.reset_index(drop=True)
    print(f"Merge {len(data)} segmentations")

    for _, row in tqdm(data.iterrows(), total=len(data)):
        # Load labels and map to new metadata specification
        file = row["image"].replace("/", "_")
        label_total = lm.transform_totals(join(config.ukbb, "preds_total", file))

        # Merge labels
        if segmentation_generation == 0:
            label_mr = lm.transform_mrs(join(config.ukbb, "preds_mr", file))
            new_label = merge_label(label_total, label_mr, channels_MRSeg_V1)
        elif segmentation_generation == 1:
            label_mr = LoadImage()(join(config.ukbb, "preds_mr2", file))
            new_label = merge_label(label_total, label_mr, channels_MRSeg_V2)
        else:
            raise Exception("Unknow segmentation generation")

        # Save label
        Save("preds_combined2")(new_label)


def Clean_sections_04():
    """Clean predictions of TotalSegmentator of sections zero and four."""
    data = pd.read_csv(join(config.ukbb, "manifest.csv"))

    # remove patients with more than 6 scans? Might be erroneous?
    patients2remove = data.loc[data["section"] == 6]["eid"].unique()
    _index = data.loc[data["eid"].apply(lambda x: x in patients2remove)].index
    data = data.drop(_index).reset_index(drop=True)
    print(f"Dropped {len(patients2remove)} patients")

    # Select water only images of section 0,4
    data = data.loc[data["dixon_type"] == "W"]
    data = data.loc[data["section"].apply(lambda x: x in [0, 4])]
    data = data.reset_index(drop=True)
    print(f"Merge {len(data)} segmentations")

    for _, row in tqdm(data.iterrows(), total=len(data)):
        # Load labels and map to new metadata specification
        file = row["image"].replace("/", "_")
        try:
            label_total = lm.transform_totals(join(config.ukbb, "preds_total", file))
            if row["section"] == 0:
                label_total = lm.clean_section0(label_total)
            elif row["section"] == 4:
                label_total = lm.clean_section4(label_total)

            # save label
            Save("section04")(label_total)
        except FileNotFoundError:
            print("File not found: ", file)


def map_merge_charite():
    """Merge annotations of charite dataset with predictions of remaining channels by MRSegmentator"""
    data = pd.read_csv(config.ukbb + "csv/charite_annotations.csv")

    for _, row in tqdm(data.iterrows(), total=len(data)):
        # Load labels
        annotation = lm.transform_mrs(join(config.mr_label_path, row["label"]))
        prediction = LoadImage()(join(config.ukbb, "preds_charite", row["pred"]))

        # Merge labels
        new_label = merge_label(prediction, annotation, channels_charite)

        # save label
        Save("preds_charite_combined")(new_label)


if __name__ == "__main__":
    map_merge_charite()
