import os
from os.path import join

import pandas as pd
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Lambdad,
    LoadImaged,
    MapLabelValued,
    Spacingd,
)
from tqdm import tqdm

import mappings

spacing = (1.5, 1.5, 3.0)

convoluted_load = Compose(
    [
        LoadImaged(keys=["pred", "label"], image_only=True),
        EnsureChannelFirstd(keys=["pred", "label"]),
        Spacingd(
            keys=["pred", "label"],
            pixdim=spacing,
            mode="nearest",
        ),
        # Use correct mapping for dataset
        MapLabelValued(
            keys=["label"],
            orig_labels=mappings.amos_to_mrseg.keys(),
            target_labels=mappings.amos_to_mrseg.values(),
        ),
        AsDiscreted(keys=["pred", "label"], to_onehot=41),
        Lambdad(keys=["pred", "label"], func=lambda x: x[None]),
    ]
)


def calc_metric(data):

    dice_metric = DiceMetric(include_background=False, reduction="none", num_classes=41)
    data_list = data[["label", "pred"]].to_dict("records")

    for item in tqdm(data_list, desc='Calc. Dice"'):
        _item = convoluted_load(item)
        dice_metric(_item["pred"], _item["label"])

    scores = pd.DataFrame(dice_metric.get_buffer())
    return scores


def main():

    data_path = ""  # should have two sub directories preds and labels
    result_path = "."

    names = [f.name for f in os.scandir(join(data_path, "preds")) if f.name[-7:] == ".nii.gz"]
    names.sort()

    preds = [join(data_path, "preds", n) for n in names]
    labels = [join(data_path, "labels", n) for n in names]

    data = pd.DataFrame()
    data["pred"] = preds
    data["label"] = labels

    # Calculate Dice
    scores = calc_metric(data)

    scores.to_csv(join(result_path, "dice.csv"), index=False)


if __name__ == "__main__":
    main()
