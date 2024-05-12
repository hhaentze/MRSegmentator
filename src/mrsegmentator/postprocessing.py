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

import numpy as np
from monai.transforms import KeepLargestConnectedComponent

organ_pairs = [
    (29, 30),  # femur
    (31, 32),  # autochthonous_muscle
    (33, 34),  # iliopsoas_muscle
    (35, 36),  # gluteus maximus
    (37, 38),  # gluteus medius
    (39, 40),  # glueteus minimus
]


def remap_left_right(pred, left_class_id, right_class_id):

    if left_class_id not in pred or right_class_id not in pred:
        return pred

    # distinguish both classes
    comb_class = (pred == left_class_id) | (pred == right_class_id)
    class1 = KeepLargestConnectedComponent()(np.copy(comb_class)[None])[0]
    class2 = comb_class != class1

    # count number of voxels for each class
    left1 = ((pred == left_class_id) & class1).sum().item()
    right1 = ((pred == right_class_id) & class1).sum().item()
    left2 = ((pred == left_class_id) & class2).sum().item()
    right2 = ((pred == right_class_id) & class2).sum().item()

    # calculate probality that either class is on the left side
    prob_1_is_left = left1 / (left1 + right1 + 1)
    prob_2_is_left = left2 / (left2 + right2 + 1)

    # normalize probability
    prob_sum = prob_1_is_left + prob_2_is_left
    prob_1_is_left /= prob_sum
    prob_2_is_left /= prob_sum

    # remap classes in prediction
    pred[comb_class] = 0
    pred[class1] = left_class_id if prob_1_is_left > 0.5 else right_class_id
    pred[class2] = left_class_id if prob_2_is_left > 0.5 else right_class_id

    return pred


def remap_wrapper(pred):
    for pair in organ_pairs:
        pred = remap_left_right(pred, pair[0], pair[1])
    return pred
