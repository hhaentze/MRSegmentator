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

import unittest

from mrsegmentator import utils


class TestUtils(unittest.TestCase):

    def test_divide_chunks(self):
        l = list(range(10))
        chunks = utils.divide_chunks(l, 5)
        self.assertEqual(next(chunks), [0, 1, 2, 3, 4])
        self.assertEqual(next(chunks), [5, 6, 7, 8, 9])

        chunks = utils.divide_chunks(l, 4)
        self.assertEqual(next(chunks), [0, 1, 2, 3])
        self.assertEqual(next(chunks), [4, 5, 6, 7])
        self.assertEqual(next(chunks), [8, 9])

        chunks = utils.divide_chunks(l, 11)
        self.assertEqual(next(chunks), list(range(10)))

        chunks = utils.divide_chunks(l, 2)
        self.assertNotEqual(next(chunks), [0, 1, 2])

    def test_add_postfix(self):
        self.assertEqual(utils.add_postfix("dir/image.nii", "seg"), "dir/image_seg.nii")
        self.assertEqual(utils.add_postfix("image.nii", "seg"), "image_seg.nii")
        self.assertEqual(utils.add_postfix("image.nii.gz", "seg"), "image_seg.nii.gz")
        self.assertEqual(utils.add_postfix("image.mha", "seg"), "image_seg.mha")
        self.assertEqual(utils.add_postfix("image.nrrd", "seg"), "image_seg.nrrd")

        self.assertNotEqual(utils.add_postfix("image.nii.gz", "seg"), "image.nii.gz")

    def test_flatten(self):
        l = [[0, 1], [2, 3], [4, 5]]
        self.assertEqual(utils.flatten(l), [0, 1, 2, 3, 4, 5])

        l = [[[0]], [[1]]]
        self.assertEqual(utils.flatten(l), [[0], [1]])


if __name__ == "__main__":
    unittest.main()
