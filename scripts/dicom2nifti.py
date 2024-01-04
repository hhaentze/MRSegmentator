from os import mkdir, scandir
from os.path import exists, join
from pathlib import Path

import config
import pandas as pd
import SimpleITK as sitk
from dicom_io import read_dicom_series_zipped
from tqdm import tqdm


def main():
    """Convert dicom files to nifti and document results in manifest.csv"""

    DATA_DIR = Path(config.ukbb)
    zip_list = [f for f in scandir(join(DATA_DIR, "dicom")) if f.name[-4:] == ".zip"]

    data = pd.DataFrame(columns=["eid", "datafile", "section", "dixon_type", "image"])

    for file in tqdm(zip_list, postfix="Reading multiple patients", leave=False, position=0):
        eid = file.name.split("_")[0]
        datafile = file.name[len(eid) + 1 : -4]
        if not exists(join(DATA_DIR, "nifti", eid)):
            mkdir(join(DATA_DIR, "nifti", eid))

        images, series_desc = read_dicom_series_zipped(file, pbar_position=1)
        image_meta = pd.DataFrame({"series_desc": series_desc})
        image_meta["dixon_type"] = image_meta["series_desc"].apply(lambda x: x.split("_")[-1])

        for i, row in tqdm(
            image_meta.iterrows(),
            total=len(image_meta),
            postfix="Saving as nifti",
            leave=False,
            position=1,
        ):
            entry = {
                "eid": eid,
                "datafile": datafile,
                "section": i
                // 4,  # divide by 4, because we have 4 different sequence types for each section
                "dixon_type": row["dixon_type"],
                "image": join(eid, "section" + str(i // 4) + "_" + row["dixon_type"] + ".nii.gz"),
            }
            data.loc[len(data)] = entry
            sitk.WriteImage(images[i], join(DATA_DIR, "nifti", entry["image"]), True, 1)

        data.to_csv(join(DATA_DIR, "manifest.csv"), index=False)


if __name__ == "__main__":
    main()
