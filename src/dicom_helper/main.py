# Copyright 2025 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import logging
import sys

from dicom_helper import dicom_conversion, parser


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # default level
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:

    setup_logging()

    # initialize Parser
    namespace = parser.initialize()
    if not namespace:
        sys.exit(1)

    try:
        if namespace.command == "list":
            series_info = dicom_conversion.list_series(namespace.input)
            print(f"\nFound {len(series_info)} series in {namespace.input}:")
            for i, (sid, desc, mod, count) in enumerate(series_info):
                print(f"  {i}: {sid}")
                print(f"     Description: {desc}")
                print(f"     Modality: {mod}")
                print(f"     Files: {count}")
                print()

        elif namespace.command == "d2n":
            nifti_file = dicom_conversion.dicom_to_nifti(
                namespace.input, namespace.output, namespace.series
            )
            print(f"Conversion completed: {nifti_file}")

        elif namespace.command == "n2d":
            dicom_conversion.nifti_to_dicom_slices(
                namespace.nifti, namespace.template, namespace.output, namespace.dtype
            )
            print(f"Conversion completed: {namespace.output}")

        elif namespace.command == "n2seg":
            dicom_conversion.nifti_to_dicom_seg(
                namespace.nifti, namespace.template, namespace.output
            )
            print(f"Conversion completed: {namespace.output}")

    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
