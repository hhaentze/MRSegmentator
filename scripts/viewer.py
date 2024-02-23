# Fork of https://github.com/kbressem/trainlib/blob/main/trainlib/viewer.py

from typing import List, Optional, Sequence, Tuple, Union

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display
from ipywidgets import widgets
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.utils import ensure_tuple
from monai.utils import convert_to_numpy


class ShapeMissmatchError(Exception):
    def __init__(self, a, b):
        message = f"Shapes of x {a.shape} and y {b.shape} do not match."
        super().__init__(message)


def _create_label(text: str) -> ipywidgets.widgets.Label:
    """Create label widget"""

    label = widgets.Label(
        text,
        layout=widgets.Layout(width="100%", display="flex", justify_content="center"),
    )
    return label


def _create_slider(
    slider_min: int,
    slider_max: int,
    value: Union[Sequence[int], int],
    step: Union[int, float] = 1,
    description: str = "",
    continuous_update: bool = True,
    readout: bool = False,
    slider_type: str = "IntSlider",
    **kwargs,
) -> ipywidgets.widgets:
    """Create slider widget"""

    slider = getattr(widgets, slider_type)(
        min=slider_min,
        max=slider_max,
        step=step,
        value=value,
        description=description,
        continuous_update=continuous_update,
        readout=readout,
        layout=widgets.Layout(width="auto", min_width="200px"),
        style={"description_width": "initial"},
        **kwargs,
    )
    return slider


def _create_button(description: str) -> ipywidgets.widgets.Button:
    """Create button widget"""
    button = widgets.Button(
        description=description, layout=widgets.Layout(width="95%", margin="5px 5px")
    )
    return button


def _create_togglebutton(description: str, value: int, **kwargs) -> ipywidgets.widgets.Button:
    """Create toggle button widget"""
    button = widgets.ToggleButton(
        description=description,
        value=value,
        layout=widgets.Layout(width="95%", margin="5px 5px"),
        **kwargs,
    )
    return button


def _create_dropdown(
    description: str, value: int, options: list[Union[str, int]], **kwargs
) -> ipywidgets.widgets.Dropdown:
    """Create dropdown widget"""
    dropdown = widgets.Dropdown(
        description=description,
        value=value,
        options=options,
        layout=widgets.Layout(width="95%", margin="5px 5px"),
        **kwargs,
    )
    return dropdown


class BasicViewer:
    """Base class for viewing TensorDicom3D objects.

    Args:
        x: main image object to view
        y: either a segmentation mask a label as str/number.
        start_view: plot image in axial, sagittal or coronal view
        prediction: a class prediction as str/number
        description: description of the whole image
        figsize: size of image, passed as plotting argument
        cmap: colormap for the image. Ignored if mode = 'RGB'.
        mask_alpha: set transparency of segmentation mask, if one is provided
        background_threshold: Values below this are shown as fully transparent
        mode: if `RGB` a three channel 2D image is plotted as 2d color image. Otherwise ignored
    Returns:
        Instance of BasicViewer
    """

    def __init__(
        self,
        x: NdarrayOrTensor,
        y: Optional[Union[NdarrayOrTensor, str]] = None,
        start_view: Optional[int] = 2,
        prediction: Optional[str] = None,
        description: Optional[str] = None,
        figsize: Tuple[int, int] = (3, 3),
        cmap: Optional[str] = "bone",
        mask_alpha: float = 0.25,
        background_threshold: float = 0.05,
        mode: Optional[str] = None,
    ):
        x = self._ensure_correct_dims(convert_to_numpy(x))

        if isinstance(y, (torch.Tensor, np.ndarray)):
            y = self._ensure_correct_dims(convert_to_numpy(y))

            if (x.shape[-2:] != y.shape[-2:]) or (mode != "RGB" and x.shape != y.shape):
                raise ShapeMissmatchError(x, y)
            self.with_mask = True
        else:
            if isinstance(y, (float, int)):
                y = str(y)
            self.with_mask = False

        if prediction is not None:
            prediction = str(prediction)

        self.x = x
        self.y = y
        self.start_view = start_view
        self.prediction = prediction
        self.description = description
        self.figsize = figsize
        self.cmap = cmap if mode != "RGB" else None
        self.slice_range = (1, 1) if mode == "RGB" and len(x) == 3 else (1, max(x.shape))
        self.mask_alpha = mask_alpha
        self.background_threshold = background_threshold
        self.mode = mode

    def _plot_slice(
        self, im_slice: int, with_mask: bool, px_range: Tuple[int, int], view: int
    ) -> None:
        """Plot slice of image"""
        if im_slice > self.x.shape[view]:
            print("Out of image borders")
            return
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        get_slice = lambda x, view, pos: (
            x[pos, :, :] if view == 0 else x[:, pos, :] if view == 1 else x[:, :, pos]
        )

        if self.mode == "RGB" and self.x.shape[0] == 4:
            image_slice = self.x.clip(*px_range)
        else:
            image_slice = get_slice(self.x, view, im_slice - 1).clip(*px_range)
        ax.imshow(
            image_slice,
            cmap=self.cmap,
            vmin=px_range[0],
            vmax=px_range[1],
        )
        if with_mask and isinstance(self.y, (torch.Tensor, np.ndarray)):
            image_slice = get_slice(self.y, view, im_slice - 1)
            alpha = np.zeros(image_slice.shape)
            alpha[image_slice > self.background_threshold] = self.mask_alpha
            ax.imshow(
                image_slice,
                cmap="jet",
                alpha=alpha,
                vmin=self.y.min(),
                vmax=self.y.max(),
            )
        plt.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    def _create_image_box(self) -> widgets.VBox:
        """Create widget items, order them in item_box and generate view box"""
        items = []

        if self.description:
            plot_description = _create_label(self.description)

        if isinstance(self.y, str):
            label = (
                f"label: {self.y} | pred: {self.prediction}"
                if self.prediction is not None
                else self.y
            )
            y_label = _create_label(label)
        else:
            y_label = None

        slice_slider = _create_slider(
            slider_min=min(self.slice_range),
            slider_max=max(self.slice_range),
            value=1 if 1 == max(self.slice_range) else self.x.shape[self.start_view] // 2,
            readout=True,
        )

        toggle_mask_button = _create_togglebutton("Show Mask", True)

        view_dropdown = _create_dropdown("View:", self.start_view, range(len(self.x.shape)))

        range_slider = _create_slider(
            slider_min=self.x.min(),
            slider_max=self.x.max(),
            value=[self.x.min(), self.x.max()],
            slider_type=(
                "FloatRangeSlider"
                if issubclass(self.x.dtype.type, np.floating)
                else "IntRangeSlider"
            ),
            step=0.01 if issubclass(self.x.dtype.type, np.floating) else 1,
            readout=True,
        )

        image_output = widgets.interactive_output(
            f=self._plot_slice,
            controls={
                "im_slice": slice_slider,
                "with_mask": toggle_mask_button,
                "view": view_dropdown,
                "px_range": range_slider,
            },
        )

        image_output.layout.height = f"{self.figsize[0]/1.2}in"  # suppress flickering
        image_output.layout.width = f"{self.figsize[1]/1.2}in"  # suppress flickering

        if self.description:
            items.append(plot_description)
        if y_label is not None:
            items.append(y_label)
        range_slider = widgets.HBox([range_slider, view_dropdown])
        items.append(range_slider)
        items.append(image_output)
        if isinstance(self.y, (torch.Tensor, np.ndarray)):
            slice_slider = widgets.HBox([slice_slider, toggle_mask_button])
        items.append(slice_slider)

        image_box = widgets.VBox(
            items,
            layout=widgets.Layout(border="none", margin="10px 5px 0px 0px", padding="5px"),
        )

        return image_box

    def _ensure_correct_dims(self, array: np.ndarray) -> np.ndarray:
        array = np.squeeze(array)
        if array.ndim == 2:
            array = np.expand_dims(array, 0)  # add channel
        if array.ndim != 3:
            raise ValueError(f"Input array needs to be 2d or 3d, but is {array.ndim}d")
        return array

    def _generate_views(self) -> None:
        image_box = self._create_image_box()
        self.box = widgets.HBox(children=[image_box])

    @property
    def image_box(self) -> widgets.VBox:
        return self._create_image_box()

    def show(self) -> None:
        """Shows plot using ipywidgets"""
        self._generate_views()
        plt.style.use("default")
        display(self.box)


class DicomExplorer(BasicViewer):
    """DICOM viewer for basic image analysis inside iPython notebooks.
    Can display a single 2D image or 3D volume together with a segmentation mask, a histogram
    of voxel/pixel values and some summary statistics.
    Allows simple windowing by clipping the pixel/voxel values to a region, which
    can be manually specified.

    """

    vbox_layout = widgets.Layout(
        margin="10px 5px 5px 5px",
        padding="5px",
        display="flex",
        flex_flow="column",
        align_items="center",
        min_width="250px",
    )

    def _plot_hist(self, px_range: Tuple[int, int]) -> None:
        """Create a simple histogram of pixel/voxel values"""
        x = self.x.flatten()
        fig, ax = plt.subplots(figsize=self.figsize)
        _, bins, patches = plt.hist(x, 100, color="grey")
        lwr = max(x.min(), px_range[0])
        upr = min(x.max(), px_range[1])

        for i, value in enumerate(bins[: len(patches)]):
            if value < lwr or value > upr:
                patches[i].set_facecolor("grey")
            else:
                patches[i].set_facecolor("darkblue")

        plt.show()

    def _image_summary(self, px_range: Tuple[int, int]) -> None:
        """Print basic summary statistics about pixel/voxel values"""
        x = self.x.clip(*px_range)
        diffs = x - x.mean()
        var = np.mean(np.power(diffs, 2.0))
        std = np.power(var, 0.5)
        zscores = diffs / std
        skews = np.mean(np.power(zscores, 3.0))
        kurt = np.mean(np.power(zscores, 4.0)) - 3.0

        table = (
            "Statistics:\n"
            + f"  Mean px value:     {x.mean()} \n"
            + f"  Std of px values:  {x.std()} \n"
            + f"  Min px value:      {x.min()} \n"
            + f"  Max px value:      {x.max()} \n"
            + f"  Median px value:   {np.median(x)} \n"
            + f"  Skewness:          {skews} \n"
            + f"  Kurtosis:          {kurt} \n\n"
            + "Tensor properties \n"
            + f"  Tensor shape:      {tuple(x.shape)}\n"
            + f"  Tensor dtype:      {x.dtype}"
        )
        print(table)

    def _generate_views(self) -> None:
        """Prepares and arranges all ipywidgets for presentation."""

        slice_slider = _create_slider(
            slider_min=min(self.slice_range),
            slider_max=max(self.slice_range),
            value=max(self.slice_range) // 2,
            readout=True,
        )

        toggle_mask_button = _create_togglebutton("Show Mask", True)

        range_slider = _create_slider(
            slider_min=self.x.min(),
            slider_max=self.x.max(),
            value=[self.x.min(), self.x.max()],
            continuous_update=False,
            slider_type=(
                "FloatRangeSlider"
                if issubclass(self.x.dtype.type, np.floating)
                else "IntRangeSlider"
            ),
            step=0.01 if issubclass(self.x.dtype.type, np.floating) else 1,
        )

        image_output = widgets.interactive_output(
            f=self._plot_slice,
            controls={
                "im_slice": slice_slider,
                "with_mask": toggle_mask_button,
                "px_range": range_slider,
            },
        )

        image_output.layout.height = f"{self.figsize[0]/1.2}in"  # suppress flickering
        image_output.layout.width = f"{self.figsize[1]/1.2}in"  # suppress flickering

        if self.y is not None:
            slice_slider = widgets.HBox([slice_slider, toggle_mask_button])

        hist_output = widgets.interactive_output(
            f=self._plot_hist, controls={"px_range": range_slider}
        )

        hist_output.layout.height = f"{self.figsize[0]/1.2}in"  # suppress flickering
        hist_output.layout.width = f"{self.figsize[1]/1.2}in"  # suppress flickering

        toggle_mask_button = _create_togglebutton("Show Mask", True)

        table_output = widgets.interactive_output(
            f=self._image_summary, controls={"px_range": range_slider}
        )

        table_box = widgets.VBox([table_output], layout=self.vbox_layout)

        hist_box = widgets.VBox([hist_output, range_slider], layout=self.vbox_layout)

        image_box = widgets.VBox([image_output, slice_slider], layout=self.vbox_layout)

        self.box = widgets.HBox(
            [image_box, hist_box, table_box],
            layout=widgets.Layout(
                border="solid 1px lightgrey",
                margin="10px 5px 0px 0px",
                padding="5px",
                width=f"{self.figsize[1]*2 + 3}in",
            ),
        )


class ListViewer:
    """Display multiple images with their masks or labels/predictions.
    Arguments:
        x: Tensor objects to view
        y: Tensor objects (in case of segmentation task) or class labels as string.
        prediction: Class predictions
        description: description of the whole image
        figsize: size of image, passed as plotting argument
        cmap: colormap for display of `x`
        max_n: maximum number of items to display
    """

    def __init__(
        self,
        x: Union[List[NdarrayOrTensor], Tuple],
        y: Optional[List[Union[NdarrayOrTensor, str]]] = None,
        start_view: Optional[int] = 2,
        prediction: Optional[List[str]] = None,
        description: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (4, 4),
        cmap: Optional[str] = "bone",
        max_n: int = 9,
        mode: Optional[str] = None,
    ):
        x = ensure_tuple(x, wrap_array=True)
        self.slice_range = (1, len(x))
        if y is not None:
            y = ensure_tuple(y, wrap_array=True)  # type: ignore
            if len(x) != len(y):  # type: ignore
                raise ValueError(f"Number of images ({len(x)}) and labels ({len(y)}) doesn't match")  # type: ignore
            y = y[0:max_n]  # type: ignore
        x = x[0:max_n]

        self.x = x
        self.y = y
        self.start_view = start_view
        self.prediction = prediction
        self.description = description
        self.figsize = figsize
        self.cmap = cmap if mode != "RGB" else None
        self.max_n = max_n
        self.mode = mode

    def _generate_views(self) -> None:
        """Arranges and prepares all ipywidgets for presentation"""
        n_images = len(self.x)
        image_grid, image_list = [], []

        for i in range(0, n_images):
            image = self.x[i]
            mask = self.y[i] if isinstance(self.y, (tuple, list)) else None
            prediction = self.prediction[i] if self.prediction else None
            description = self.description[i] if self.description else None

            image_list.append(
                BasicViewer(
                    x=image,
                    y=mask,
                    start_view=self.start_view,
                    prediction=prediction,
                    description=description,
                    figsize=self.figsize,
                    cmap=self.cmap,
                    mode=self.mode,
                ).image_box
            )

            if (i + 1) % np.ceil(np.sqrt(n_images)) == 0 or i == n_images - 1:
                image_grid.append(widgets.HBox(image_list))
                image_list = []

        self.box = widgets.VBox(children=image_grid)

    def show(self) -> None:
        """Display plots using ipywidgets"""
        self._generate_views()
        plt.style.use("default")
        display(self.box)
