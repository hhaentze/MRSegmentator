from monai.transforms import Compose, LoadImage, MapLabelValue

# Create list of updated labels
new_labels = [
    # Abdmonial key organs
    "background",
    "spleen",
    "right_kidney",
    "left_kidney",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "right_adrenal_gland",
    "left_adrenal_gland",
    # Chest region
    "left_lung",  # New Entry
    "right_lung",  # New Entry
    # Blood vessels
    "heart",  # New Entry
    "aorta",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "left_iliac_artery",  # New Entry
    "right_iliac_artery",  # New Entry
    "left_iliac_vena",  # New Entry
    "right_iliac_vena",  # New Entry
    # Digestive organs
    "esophagus",
    "small_bowel",
    "duodenum",
    "colon",
    "urinary_bladder",  # New Entry
    # Bones
    "spine",  # New Entry
    "sacrum",  # New Entry
    "left_hip",  # New Entry
    "right_hip",  # New Entry
    "left_femur",  # New Entry
    "right_femur",  # New Entry
    # Muscles
    "left_autochthonous_muscle",
    "right_autochthonous_muscle",
    "left_iliopsoas_muscle",
    "right_iliopsoas_muscle",
    "left_gluteus_maximus",  # New Entry
    "right_gluteus_maximus",  # New Entry
    "left_gluteus_medius",  # New Entry
    "right_gluteus_medius",  # New Entry
    "left_gluteus_minimus",  # New Entry
    "right_gluteus_minimus",  # New Entry
]
new_labels = {i: label for i, label in enumerate(new_labels)}


# Create new Mapppings
mr2new = [0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 7, 8, 9, 20, 21, 22, 23, 31, 32, 33, 34]
mr2new = {i: label for i, label in enumerate(mr2new)}

total2new = list(range(10))
total2new += [10, 10, 11, 11, 11]  # lung lobes
total2new += [20, 0, 0]  # esophagus
total2new += [21, 22, 23, 24]  # digestive
total2new += [0, 0, 0]
total2new += [26] + [25] * 25  # sacrum + spine
total2new += [
    12,
    13,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    14,
    15,
    16,
    17,
    18,
    19,
]  # blood vessels
total2new += [0] * 6
total2new += [29, 30, 27, 28]  # bones
total2new += [0]
total2new += [35, 36, 37, 38, 39, 40, 31, 32, 33, 34]  # muscles
total2new += [0] * 28
total2new = {i: label for i, label in enumerate(total2new)}

# Apply mappings
transform_total = MapLabelValue(
    orig_labels=total2new.keys(),
    target_labels=total2new.values(),
    dtype=int,
)
transform_mr = MapLabelValue(
    orig_labels=mr2new.keys(),
    target_labels=mr2new.values(),
    dtype=int,
)
transform_totals = Compose((LoadImage(image_only=True), transform_total))
transform_mrs = Compose((LoadImage(image_only=True), transform_mr))


clean_section0 = MapLabelValue(
    orig_labels=[21, 22, 23, 24] + [26, 27, 28, 29, 30] + [33, 34, 35, 36, 37, 38, 39, 40],
    target_labels=[0] * (4 + 5 + 8),
    dtype=int,
)

clean_section4 = MapLabelValue(
    orig_labels=[1] + [4, 5, 6, 7] + [10, 11, 12] + [15] + [20],
    target_labels=[0] * (1 + 4 + 3 + 1 + 1),
    dtype=int,
)
