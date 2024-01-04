from os.path import join

import config
import labelmappings as lm
import pandas as pd
from monai.transforms import KeepLargestConnectedComponent, MapLabelValue, SaveImage
from tqdm import tqdm

deleteMusclesGall = MapLabelValue(
    orig_labels=[4, 31, 32, 33, 34],
    target_labels=[0, 0, 0, 0, 0],
    dtype=int,
)
deleteAllButMusclesGall = MapLabelValue(
    orig_labels=list(range(4)) + list(range(5, 31)) + list(range(35, 41)),
    target_labels=[0] * 36,
    dtype=int,
)

keepLargestMusclesGall = KeepLargestConnectedComponent(applied_labels=[4, 31, 32, 33, 34])

save = SaveImage(
    output_dir=join(config.ukbb, "preds_combined"),
    separate_folder=False,
    output_postfix="new",
    print_log=False,
)


def merge_label(totalseg_label, mrseg_label):
    # clean labels
    totalseg_label = deleteMusclesGall(totalseg_label)
    mrseg_label = deleteAllButMusclesGall(mrseg_label)

    # merge labels
    new_label = mrseg_label.detach().clone()
    new_label[totalseg_label != 0] = totalseg_label[totalseg_label != 0]

    # remove artifacts that were created during merging
    new_label = keepLargestMusclesGall(new_label)
    return new_label


def main():
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
        label_mr = lm.transform_mrs(join(config.ukbb, "preds_mr", file))
        label_total = lm.transform_totals(join(config.ukbb, "preds_total", file))

        # merge labels
        new_label = merge_label(label_total, label_mr)

        # save label
        save(new_label)


def map_s0_s4():
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
            save(label_total)
        except FileNotFoundError:
            print("File not found: ", file)


if __name__ == "__main__":
    map_s0_s4()
