import os

from mrsegmentator import parser

# disable warning message about undefined environmental variables
# (We assign temporary arbitrary values. The script does not use these)
if os.environ.get("nnUNet_raw") is None:
    os.environ["nnUNet_raw"] = "empty"
if os.environ.get("nnUNet_preprocessed") is None:
    os.environ["nnUNet_preprocessed"] = "empty"
if os.environ.get("nnUNet_results") is None:
    os.environ["nnUNet_results"] = "empty"

from batchgenerators.utilities.file_and_folder_operations import join

from mrsegmentator.inference import infer


def crossval(namespace, images):
    """Run each model individually"""

    for fold in range(5):
        # make directory
        outdir = join(namespace.outdir, "fold" + str(fold))
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        # run inference
        infer(
            namespace.modeldir,
            (fold,),
            outdir,
            images,
            namespace.postfix,
            namespace.force_LPS,
            namespace.verbose,
            namespace.cpu_only,
        )


def read_images(namespace):
    # images must be of nifti format
    condition = lambda x: x[-7:] == ".nii.gz" or x[-4:] == ".nii"

    # look for images in input directory
    if os.path.isdir(namespace.input):
        images = [f.path for f in os.scandir(namespace.input) if condition(f.name)]
        assert (
            len(images) > 0
        ), f"no images with file ending .nii or .nii.gz in direcotry {namespace.input}"
    else:
        images = [namespace.input]
        assert condition(
            images[0]
        ), f"file ending of {namespace.input} neither .nii nor .nii.gz"

    return images


def main():
    # initialize Parser
    namespace = parser.initialize()
    parser.assert_namespace(namespace)

    # select images for segmentation
    images = read_images(namespace)

    # run all models individually
    if namespace.crossval:
        crossval(namespace, images)
        return

    # ensemble prediction
    if namespace.fold is None:
        folds = (
            0,
            1,
            2,
            3,
            4,
        )

    # single prediction
    else:
        folds = (namespace.fold,)

    # run inference
    infer(
        namespace.modeldir,
        folds,
        namespace.outdir,
        images,
        namespace.postfix,
        namespace.force_LPS,
        namespace.verbose,
        namespace.cpu_only,
    )


if __name__ == "__main__":
    main()