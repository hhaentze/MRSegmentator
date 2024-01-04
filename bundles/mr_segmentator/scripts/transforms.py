from typing import Dict, Hashable, Mapping, Optional, Sequence

import torch
from monai import transforms
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.utils.enums import PostFix, TransformBackends

DEFAULT_POST_FIX = PostFix.meta()


class ApplyWindowing(transforms.Transform):
    """
    Apply window presets to DICOM images
    Windowing adapts the greyscale component of a CT image to highlight particular structures
    by reducing the range of Hounsfield units (HU) to be displayed. Windows are usually defined by
    a width (ww), the range of HU to be considered and level (wl, the center of the window). A level of 50
    and width of 100 will thus clip all values to the range of 0 and 100.

    Args:
        window: a string for preset windows. Implemented presets are:
            brain: ww 80, wl 40
            subdural: ww 130, wl = 50
            stroke: ww 8, wl 40
            temporal bone: ww 2800, wl 700
            lungs: ww 150, wl -600
            abdomen: ww 400, wl 50
            liver: ww 150, wl 30
            bone: ww 1800, wl 400
        upper: upper threshold for windowing
        lower: lower threshold for windowing
        width: window width
        level: window level (or windo center)

    Raises:
        Either `window`, `lower`/`upper` or `width`/`level` should be specified.
        Otherwise ValueError is raised
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        window: Optional[str] = None,
        upper: Optional[int] = None,
        lower: Optional[int] = None,
        width: Optional[int] = None,
        level: Optional[int] = None,
    ):
        error_message = "Please specifiy either window or upper/lower or width/level."
        if window:
            if upper or lower:
                raise ValueError(error_message)
            if width or level:
                raise ValueError(error_message)
        elif upper and lower:
            if window:
                raise ValueError(error_message)
            if width or level:
                raise ValueError(error_message)
        elif width and level:
            if upper or lower:
                raise ValueError(error_message)
            if window:
                raise ValueError(error_message)
        else:
            raise ValueError(error_message)

        if window:
            if window == "brain":
                width, level = 80, 40
            elif window == "subdural":
                width, level = 130, 50
            elif window == "stroke":
                width, level = 8, 40
            elif window == "temporal bone":
                width, level = 2800, 700
            elif window == "lungs":
                width, level = 150, -600
            elif window == "abdomen":
                width, level = 400, 50
            elif window == "liver":
                width, level = 150, 30
            elif window == "bone":
                width, level = 1800, 400

        if width and level:
            upper = level + width // 2
            lower = level - width // 2

        self.upper = upper
        self.lower = lower

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        return img.clip(self.lower, self.upper)


class ApplyWindowingd(transforms.MapTransform):
    "Dictionary-based wrapper of :py:class:`ApplyWindowing`."

    def __init__(
        self,
        keys: KeysCollection,
        window: Optional[str] = None,
        upper: Optional[int] = None,
        lower: Optional[int] = None,
        width: Optional[int] = None,
        level: Optional[int] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.windowing = ApplyWindowing(
            window=window, upper=upper, lower=lower, width=width, level=level
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.windowing(d[key])
        return d


class RemoveOrgansAbove(transforms.Transform):
    """Remove organs above other organs"""

    def __init__(
        self,
        organs_to_remove: Sequence[int] | int,
        organs_to_keep: Sequence[int] | int,
        mode: str = "center",
    ) -> None:
        self.organs_to_remove = (
            organs_to_remove if isinstance(organs_to_remove, list) else [organs_to_remove]
        )
        self.organs_to_keep = (
            organs_to_keep if isinstance(organs_to_keep, list) else [organs_to_keep]
        )
        self.mode = mode
        if mode not in ["center", "start", "end"]:
            raise Exception(f"Unknownn mode {self.mode}")

    def __call__(self, label: NdarrayOrTensor) -> NdarrayOrTensor:
        # select bounding box
        shape = label.squeeze().shape
        start = torch.tensor(shape)
        end = torch.zeros(len(shape))
        for organ in self.organs_to_keep:
            _, _s, _e = transforms.CropForeground(
                select_fn=lambda x: x == organ, margin=0, return_coords=True, allow_smaller=False
            )(label)
            start = torch.min(start, torch.tensor(_s))
            end = torch.max(end, torch.tensor(_e))
        center = start + ((end - start) // 2)

        # choose mode
        if self.mode == "center":
            x = int(center[2].item())
        elif self.mode == "start":
            x = int(start[2].item())
        else:
            x = int(end[2].item())

        # delete organ above x
        for organ in self.organs_to_remove:
            label[:, :, :, x:][label[:, :, :, x:] == organ] = 0
        return label


class RemoveOrgansAboved(transforms.MapTransform):
    """Remove organs above other organs"""

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        **kwargs,
    ) -> None:
        self.main = RemoveOrgansAbove(**kwargs)
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.main(d[key])
        return d


class MatchSized(transforms.MapTransform):
    """Match size of item to the size of a reference item"""

    def __init__(
        self,
        keys: KeysCollection,
        reference_key: str,
        mode: str,
        allow_missing_keys: bool = False,
        **kwargs,
    ) -> None:
        self.mode = mode
        self.reference_key = reference_key
        self.resize_kwargs = kwargs
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        reference = d[self.reference_key]
        resize = transforms.Resize(
            reference.shape[1:],  # assumes channel first format
            mode=self.mode,
            **self.resize_kwargs,
        )
        for key in self.key_iterator(d):
            d[key] = resize(d[key])
        return d


class RestoreOriginalSpacing(transforms.MapTransform):
    """Match size of item to the size of a reference item"""

    def __init__(
        self,
        keys: KeysCollection,
        mode: str,
        allow_missing_keys: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.sp_resample = transforms.SpatialResample(mode=mode, **kwargs)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.sp_resample(d[key], dst_affine=d[key].meta["original_affine"])
        return d


class RemoveSmallClasses(transforms.Transform):
    """
    Remove classes with exceptionally small volumes.

    Data should be one-hotted.

    Args:
        applied_labels: Labels for applying the analysis on
        min_size: objects smaller than this size (in pixel) are removed. Must have same length as ``applied_labels``
        exclude_edge: Whether to exclude classes located at the edge of a scan. These cases might have a small volume
                    because they are not completly inside the scan. Default to true.

    Raises:
        ValueError: When ``len(applied_labels) != len(min_size)``.
    """

    def __init__(
        self,
        applied_labels: Sequence[int] | int,
        min_size: Sequence[int] | int,
        exclude_edge: bool = True,
        verbose: bool = False,
    ) -> None:
        self.applied_labels = (
            applied_labels if isinstance(applied_labels, list) else [applied_labels]
        )
        self.min_size = min_size if isinstance(min_size, list) else [min_size]
        self.exclude_edge = exclude_edge
        self.verbose = verbose

        if len(self.applied_labels) != len(self.min_size):
            raise ValueError(
                "``applied_labels`` and ``min_size`` must have same length, "
                + f"but have length of {len(self.applied_labels)} and {len(self.min_size)}."
            )

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: shape must be (C, spatial_dim1[, spatial_dim2, spatial_dim3]). Data
                should be one-hotted.

        Returns:
            An array with shape (C, spatial_dim1[, spatial_dim2, spatial_dim3]).
        """
        for label, size in zip(self.applied_labels, self.min_size):
            if self.exclude_edge and self._surface_empty(img[label]):
                if torch.sum(img[label]) <= size:
                    if self.verbose:
                        print(
                            f"RemoveSmallClasses: Removed label {label} with volume of {int(torch.sum(img[label]))}"
                            + f", which is smaller than {size}."
                        )
                    img[label] = 0

        return img

    def _surface_empty(self, img: NdarrayOrTensor, value: int = 1) -> bool:
        """check if the edges of an image are free from a specific label, i.e. `value`"""

        label_on_surface = False

        label_on_surface += value in img[0] or value in img[-1]
        if len(img.shape) >= 2:
            label_on_surface += value in img[:, 0] or value in img[:, -1]
        if len(img.shape) >= 3:
            label_on_surface += value in img[:, :, 0] or value in img[:, :, -1]
        if len(img.shape) >= 4:
            raise ValueError("Spatial dimension of input must not be higher than 3.")

        return not label_on_surface


class RemoveSmallClassesd(transforms.MapTransform):
    """Dictionary-based wrapper of :py:class:`RemoveSmallClasses`."""

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.removeSmallClasses = RemoveSmallClasses(**kwargs)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.removeSmallClasses(d[key])
        return d
