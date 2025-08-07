# Copyright 2025 Hartmut Häntze

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from typing import Any


def initialize() -> Any:

    name = "MRSegmentator DICOM Helper"
    desc = "DICOM/NIfTI conversion tool"
    epilog = "Charité AG KI - 2024"

    parser = argparse.ArgumentParser(prog=name, description=desc, epilog=epilog)

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # List series command
    p0 = sub.add_parser("list", help="List DICOM series in directory")
    p0.add_argument("--input", "-i", required=True, help="DICOM directory")

    # DICOM to NIfTI
    p1 = sub.add_parser("d2n", help="Convert DICOM to NIfTI")
    p1.add_argument("--input", "-i", required=True, help="DICOM directory")
    p1.add_argument("--output", "-o", default=None, help="Output directory")
    p1.add_argument("--series", "-s", default=None, help="Specific series ID")

    # NIfTI to DICOM slices
    p2 = sub.add_parser("n2d", help="Convert NIfTI to DICOM slices")
    p2.add_argument("--nifti", "-n", required=True, help="Input NIfTI file")
    p2.add_argument("--template", "-t", required=True, help="Template DICOM directory")
    p2.add_argument("--output", "-o", required=True, help="Output directory")
    p2.add_argument("--dtype", default="uint8", help="Output data type (default: uint8)")

    # NIfTI to DICOM SEG
    p3 = sub.add_parser("n2seg", help="Convert NIfTI to DICOM SEG")
    p3.add_argument("--nifti", "-n", required=True, help="Input NIfTI file")
    p3.add_argument("--template", "-t", required=True, help="Template DICOM directory")
    p3.add_argument("--output", "-o", required=True, help="Output DICOM SEG file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    return args
