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

from mrsegmentator import config, parser, utils

config.disable_nnunet_path_warnings()

from mrsegmentator.inference import infer  # noqa: E402


def main() -> None:
    # initialize Parser
    namespace = parser.initialize()
    parser.assert_namespace(namespace)

    # select images for segmentation
    images = utils.read_images(namespace)

    # ensemble/single prediction
    if namespace.fold is None:
        folds = (0, 1, 2, 3, 4)
    else:
        folds = (namespace.fold,)  # type: ignore

    # run inference
    infer(
        images,
        namespace.outdir,
        folds,
        namespace.postfix,
        namespace.split_level,
        namespace.verbose,
        namespace.cpu_only,
        namespace.batchsize,
        namespace.nproc,
        namespace.nproc_export,
    )


if __name__ == "__main__":
    print("test", __file__, __name__)
    main()
