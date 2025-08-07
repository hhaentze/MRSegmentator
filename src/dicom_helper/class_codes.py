# Copyright 2025 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

from highdicom.sr.coding import CodedConcept

CLASS_CODES = {
    1: CodedConcept(value="78961009", scheme_designator="SCT", meaning="Spleen"),
    2: CodedConcept(value="64033007", scheme_designator="SCT", meaning="Right kidney"),
    3: CodedConcept(value="64033007", scheme_designator="SCT", meaning="Left kidney"),
    4: CodedConcept(value="28273000", scheme_designator="SCT", meaning="Gallbladder"),
    5: CodedConcept(value="10200004", scheme_designator="SCT", meaning="Liver"),
    6: CodedConcept(value="69695003", scheme_designator="SCT", meaning="Stomach"),
    7: CodedConcept(value="15776009", scheme_designator="SCT", meaning="Pancreas"),
    8: CodedConcept(value="23451007", scheme_designator="SCT", meaning="Right adrenal gland"),
    9: CodedConcept(value="23451007", scheme_designator="SCT", meaning="Left adrenal gland"),
    10: CodedConcept(value="39607008", scheme_designator="SCT", meaning="Left lung"),
    11: CodedConcept(value="39607008", scheme_designator="SCT", meaning="Right lung"),
    12: CodedConcept(value="80891009", scheme_designator="SCT", meaning="Heart"),
    13: CodedConcept(value="15825003", scheme_designator="SCT", meaning="Aorta"),
    14: CodedConcept(value="64131007", scheme_designator="SCT", meaning="Inferior vena cava"),
    15: CodedConcept(
        value="32764006", scheme_designator="SCT", meaning="Portal vein and splenic vein"
    ),
    16: CodedConcept(value="244411005", scheme_designator="SCT", meaning="Left iliac artery"),
    17: CodedConcept(value="244411005", scheme_designator="SCT", meaning="Right iliac artery"),
    18: CodedConcept(value="113262008", scheme_designator="SCT", meaning="Left iliac vein"),
    19: CodedConcept(value="113262008", scheme_designator="SCT", meaning="Right iliac vein"),
    20: CodedConcept(value="32849002", scheme_designator="SCT", meaning="Esophagus"),
    21: CodedConcept(value="30315005", scheme_designator="SCT", meaning="Small bowel"),
    22: CodedConcept(value="38848004", scheme_designator="SCT", meaning="Duodenum"),
    23: CodedConcept(value="71854001", scheme_designator="SCT", meaning="Colon"),
    24: CodedConcept(value="89837001", scheme_designator="SCT", meaning="Urinary bladder"),
    25: CodedConcept(value="421060004", scheme_designator="SCT", meaning="Spine"),
    26: CodedConcept(value="91609006", scheme_designator="SCT", meaning="Sacrum"),
    27: CodedConcept(value="22356005", scheme_designator="SCT", meaning="Left hip"),
    28: CodedConcept(value="22356005", scheme_designator="SCT", meaning="Right hip"),
    29: CodedConcept(value="71341001", scheme_designator="SCT", meaning="Left femur"),
    30: CodedConcept(value="71341001", scheme_designator="SCT", meaning="Right femur"),
    31: CodedConcept(
        value="181310005", scheme_designator="SCT", meaning="Left autochthonous muscle"
    ),
    32: CodedConcept(
        value="181310005", scheme_designator="SCT", meaning="Right autochthonous muscle"
    ),
    33: CodedConcept(value="72107004", scheme_designator="SCT", meaning="Left iliopsoas muscle"),
    34: CodedConcept(value="72107004", scheme_designator="SCT", meaning="Right iliopsoas muscle"),
    35: CodedConcept(value="36005007", scheme_designator="SCT", meaning="Left gluteus maximus"),
    36: CodedConcept(value="36005007", scheme_designator="SCT", meaning="Right gluteus maximus"),
    37: CodedConcept(value="46813001", scheme_designator="SCT", meaning="Left gluteus medius"),
    38: CodedConcept(value="46813001", scheme_designator="SCT", meaning="Right gluteus medius"),
    39: CodedConcept(value="110568007", scheme_designator="SCT", meaning="Left gluteus minimus"),
    40: CodedConcept(value="110568007", scheme_designator="SCT", meaning="Right gluteus minimus"),
}
