from collections.abc import Callable

import torch
from monai.handlers.ignite_metric import IgniteMetric
from monai.metrics import CumulativeIterationMetric
from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction


class VolumeMetric(IgniteMetric):
    """
    Returns volume of segmented classes
    """

    def __init__(
        self,
        include_background: bool = True,
        reduction: MetricReduction | str = MetricReduction.SUM,
        num_classes: int | None = None,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
        check_empty: bool = False,
    ) -> None:
        """

        Args:
            include_background: whether to include dice computation on the first channel of the predicted output.
                Defaults to True.
            reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
            num_classes: number of input channels (always including the background). When this is None,
                ``y_pred.shape[1]`` will be used. This option is useful when both ``y_pred`` and ``y`` are
                single-channel class indices and the number of classes is not automatically inferred from data.
            output_transform: callable to extract `y_pred` and `y` from `ignite.engine.state.output` then
                construct `(y_pred, y)` pair, where `y_pred` and `y` can be `batch-first` Tensors or
                lists of `channel-first` Tensors. the form of `(y_pred, y)` is required by the `update()`.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            save_details: whether to save metric computation details per image, for example: mean dice of every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.
            check_empty: return negative volume if the groundtruth is empty

        See also:
            :py:meth:`monai.metrics.meandice.compute_dice`
        """
        metric_fn = VolumeMetricFuntion(
            reduction=reduction,
            num_classes=num_classes,
            check_empty=check_empty,
        )
        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            save_details=save_details,
        )


class VolumeMetricFuntion(CumulativeIterationMetric):
    """
    Compute volume of segmented classes in `y_pred`. If `check_empty` is true compare segmenation to `y_pred`
    and return negative volume for all classes that are absent from `y_pred`.

    It supports both multi-classes and multi-labels tasks.
    Input `y_pred` is compared with ground truth `y`.
    `y_pred` is expected to have binarized predictions and `y` can be single-channel class indices or in the
    one-hot format. The `include_background` parameter can be set to ``False`` to exclude
    the first category (channel index 0) which is by convention assumed to be background. If the non-background
    segmentations are small compared to the total image size they can get overwhelmed by the signal from the
    background. `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]),
    `y` can also be in the format of `B1HW[D]`.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        ignore_empty: whether to ignore empty ground truth cases during calculation.
            If `True`, NaN value will be set for empty ground truth cases.
            If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.
        num_classes: number of input channels (always including the background). When this is None,
            ``y_pred.shape[1]`` will be used. This option is useful when both ``y_pred`` and ``y`` are
            single-channel class indices and the number of classes is not automatically inferred from data.
        check_empty: return negative volume if the groundtruth is empty

    """

    def __init__(
        self,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        ignore_empty: bool = True,
        num_classes: int | None = None,
        check_empty: bool = False,
    ) -> None:
        super().__init__()
        self.include_background = False
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.ignore_empty = ignore_empty
        self.num_classes = num_classes
        self.check_empty = check_empty

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute mean Dice metric. `y` can be single-channel class indices or
                in the one-hot format.

        Raises:
            ValueError: when `y_pred` has less than three dimensions.
        """
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(
                f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}."
            )

        # compute dice (BxC) for each channel for each batch
        if self.num_classes is None:
            n_pred_ch = y_pred.shape[1]  # y_pred is in one-hot format or multi-channel scores
        else:
            n_pred_ch = self.num_classes

        data = []
        for b in range(y_pred.shape[0]):
            c_list = []
            for c in range(1, n_pred_ch) if n_pred_ch > 1 else [1]:
                x_pred = (y_pred[b, 0] == c) if (y_pred.shape[1] == 1) else y_pred[b, c].bool()
                x = (y[b, 0] == c) if (y.shape[1] == 1) else y[b, c]
                if self.check_empty:
                    c_list.append(torch.sum(x_pred) if 1 in x else -torch.sum(x_pred))
                else:
                    c_list.append(torch.sum(x_pred))
            data.append(torch.stack(c_list))
        data = torch.stack(data, dim=0).contiguous()

        return data

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reduction and aggregation logic for the output of `compute_dice`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"the data to aggregate must be PyTorch Tensor, got {type(data)}.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f
