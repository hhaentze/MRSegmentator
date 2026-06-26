# tests/integration_config.py
#
# Configuration for the integration test suite.
# Copy this file, remove the .example suffix, and fill in the paths below.
# integration_config.py is intentionally excluded from version control.
#
# Run:
#   make smoke   # fast, no model required
#   make full    # smoke + integration (requires weights and a real image)

from pathlib import Path

# Directory that contains the nnUNet_results/ subfolder (main model weights).
MODEL_PATH = Path("/path/to/weights")

# Optional: path to body-composition model weights (future --body-comp flag).
# Leave as None to skip body-comp tests.
BODY_COMP_MODEL_PATH = None

# One or more real MRI/CT files (.nii, .nii.gz, .mha, or .nrrd).
# All images are passed to a single infer() call, so inference is initialised
# only once regardless of list length.  Add more files for broader coverage;
# remove them to keep runtime short.
TEST_IMAGES = [Path("/path/to/image.nii.gz")]
