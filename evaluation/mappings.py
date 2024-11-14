# Copyright 2024 Hartmut Häntze

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Helper File with importan channels and mapping information """


total_to_mrseg = {
    1: 1,  # spleen
    2: 2,  # kidney_right -> right_kidney
    3: 3,  # kidney_left -> left_kidney
    4: 4,  # gallbladder
    5: 5,  # liver
    6: 6,  # stomach
    7: 7,  # pancreas
    8: 8,  # adrenal_gland_right -> right_adrenal_gland
    9: 9,  # adrenal_gland_left -> left_adrenal_gland
    10: 10,  # lung_left -> left_lung
    11: 11,  # lung_right -> right_lung
    12: 20,  # esophagus
    13: 21,  # small_bowel -> small_bowel
    14: 22,  # duodenum
    15: 23,  # colon
    16: 24,  # urinary_bladder
    17: 0,  # prostate -> Not in MRSegmentator
    18: 26,  # sacrum
    19: 25,  # vertebrae -> spine
    20: 0,  # intervertebral_discs -> Not in MRSegmentator
    21: 0,  # spinal_cord -> Not in MRSegmentator
    22: 12,  # heart
    23: 13,  # aorta
    24: 14,  # inferior_vena_cava
    25: 15,  # portal_vein_and_splenic_vein
    26: 16,  # iliac_artery_left -> left_iliac_artery
    27: 17,  # iliac_artery_right -> right_iliac_artery
    28: 18,  # iliac_vena_left -> left_iliac_vena
    29: 19,  # iliac_vena_right -> right_iliac_vena
    30: 0,  # humerus_left -> Not in MRSegmentator
    31: 0,  # humerus_right -> Not in MRSegmentator
    32: 0,  # fibula -> Not in MRSegmentator
    33: 0,  # tibia -> Not in MRSegmentator
    34: 29,  # femur_left -> left_femur
    35: 30,  # femur_right -> right_femur
    36: 27,  # hip_left -> left_hip
    37: 28,  # hip_right -> right_hip
    38: 35,  # gluteus_maximus_left -> left_gluteus_maximus
    39: 36,  # gluteus_maximus_right -> right_gluteus_maximus
    40: 37,  # gluteus_medius_left -> left_gluteus_medius
    41: 38,  # gluteus_medius_right -> right_gluteus_medius
    42: 39,  # gluteus_minimus_left -> left_gluteus_minimus
    43: 40,  # gluteus_minimus_right -> right_gluteus_minimus
    44: 31,  # autochthon_left -> left_autochthonous_muscle
    45: 32,  # autochthon_right -> right_autochthonous_muscle
    46: 33,  # iliopsoas_left -> left_iliopsoas_muscle
    47: 34,  # iliopsoas_right -> right_iliopsoas_muscle
    48: 0,  # quadriceps_femoris_left -> Not in MRSegmentator
    49: 0,  # quadriceps_femoris_right -> Not in MRSegmentator
    50: 0,  # thigh_medial_compartment_left -> Not in MRSegmentator
    51: 0,  # thigh_medial_compartment_right -> Not in MRSegmentator
    52: 0,  # thigh_posterior_compartment_left -> Not in MRSegmentator
    53: 0,  # thigh_posterior_compartment_right -> Not in MRSegmentator
    54: 0,  # sartorius_left -> Not in MRSegmentator
    55: 0,  # sartorius_right -> Not in MRSegmentator
    56: 0,  # brain -> Not in MRSegmentator
}

amos_to_mrseg = {
    1: 1,  # spleen
    2: 2,  # right kidney
    3: 3,  # left kidney
    4: 4,  # gall bladder
    5: 20,  # esophagus
    6: 5,  # liver
    7: 6,  # stomach
    8: 13,  # aorta
    9: 14,  # postcava
    10: 7,  # pancreas
    11: 8,  # right adrenal gland
    12: 9,  # left adrenal gland
    13: 22,  # duodenum
    14: 0,  # urinary bladder
    15: 0,  # uterus / prostate
}


channels = [
    "spleen",
    "right_kidney",
    "left_kidney",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "right_adrenal_gland",
    "left_adrenal_gland",
    "left_lung",
    "right_lung",
    "heart",
    "aorta",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "left_iliac_artery",
    "right_iliac_artery",
    "left_iliac_vena",
    "right_iliac_vena",
    "esophagus",
    "small_bowel",
    "duodenum",
    "colon",
    "urinary_bladder",
    "spine",
    "sacrum",
    "left_hip",
    "right_hip",
    "left_femur",
    "right_femur",
    "left_autochthonous_muscle",
    "right_autochthonous_muscle",
    "left_iliopsoas_muscle",
    "right_iliopsoas_muscle",
    "left_gluteus_maximus",
    "right_gluteus_maximus",
    "left_gluteus_medius",
    "right_gluteus_medius",
    "left_gluteus_minimus",
    "right_gluteus_minimus",
]


""" Classes that are completly within the field of view in our T2-HASTE images"""
T2_channels = [
    "spleen",
    "right_kidney",
    "left_kidney",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "right_adrenal_gland",
    "left_adrenal_gland",
    "left_lung",
    "right_lung",
    "heart",
    "aorta",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "esophagus",
    "small_bowel",
    "duodenum",
    "colon",
    "spine",
    "left_autochthonous_muscle",
    "right_autochthonous_muscle",
    "left_iliopsoas_muscle",
    "right_iliopsoas_muscle",
]
