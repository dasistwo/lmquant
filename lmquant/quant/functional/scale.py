# -*- coding: utf-8 -*-
"""Quantization scale module."""

import gc
import math
import torch

from ...dataset import ActivationsCache
from ..data.dtype import QuantDataType
from ..data.range import DynamicRange, QuantRange
from ..data.scale import QuantScale
from ..data.utils import ScaleUtils, ShapeUtils
from ..data.utils.scale import infer_scale_quant_spans
from .simple import simple_quantize
from .config import QuantGPTQConfig, QuantDecoupleQConfig

__all__ = ["infer_scale_and_zero", "calibrate_scale_and_zero"]


def quantize_scale_kernel(
    s: torch.Tensor,
    /,
    *,
    quant_dtypes: list[QuantDataType],
    quant_spans: list[float],
    view_shapes: list[tuple[int]],
) -> QuantScale:
    """Quantize the scale tensor.

    Args:
        s (torch.Tensor): The scale tensor.
        quant_dtypes (list[QuantDataType]): The quantization dtypes of the scale tensor.
        quant_spans (list[float]): The quantization spans of the scale tensor.
        view_shapes (list[tuple[int]]): The view shapes of the scale tensor.

    Returns:
        QuantScale: The quantized scale tensor.
    """
    scale = QuantScale()
    s = s.abs()
    for view_shape, quant_dtype, quant_span in zip(view_shapes[:-1], quant_dtypes[:-1], quant_spans[:-1]):
        s = s.view(view_shape)  # (#g0, rs0, #g1, rs1, #g2, rs2, ...)
        ss = s.amax(dim=list(range(1, len(view_shape), 2)), keepdim=True)
        ss = simple_quantize(ss / quant_span, quant_dtype=quant_dtype)
        s = s / ss
        scale.append(ss)
    view_shape = view_shapes[-1]
    s = s.view(view_shape)
    if any(v != 1 for v in view_shape[1::2]):
        ss = s.amax(dim=list(range(1, len(view_shape), 2)), keepdim=True)
        ss = simple_quantize(ss / quant_spans[-1], quant_dtype=quant_dtypes[-1])
    else:
        assert quant_spans[-1] == 1, "The last quant span must be 1."
        ss = simple_quantize(s, quant_dtype=quant_dtypes[-1])
    scale.append(ss)
    scale.data[scale.data == 0] = 1
    return scale


def infer_scale_and_zero(  # noqa: C901
    tensor: torch.Tensor,
    *,
    quant_dtype: QuantDataType,
    group_shapes: list[torch.Size],
    scale_quant_dtypes: list[QuantDataType | torch.dtype],
    exponent_scaling_level: int = None,
    view_shape: torch.Size = None,
    dynamic_range: DynamicRange = None,
    quant_range: QuantRange = None,
    scale: torch.Tensor = None,
    zero: torch.Tensor = None,
) -> tuple[QuantScale, torch.Tensor]:
    """Get the quantization scale and zero point of the tensor to be quantized.

    Args:
        tensor (torch.Tensor): The tensor to be quantized.
        quant_dtype (QuantDataType): The quantization dtype.
        group_shapes (list[torch.Size]): The group shapes for quantization.
        scale_quant_dtypes (list[QuantDataType | torch.dtype]): The quantization dtypes of the scale tensor.
        exponent_scaling_level (int, optional): The exponent scaling level. Defaults to ``None``.
        view_shape (torch.Size, optional): The view shape of the tensor. Defaults to ``None``.
        dynamic_range (DynamicRange): The dynamic range of the tensor.
        quant_range (QuantRange): The quantization range.
        scale (torch.Tensor, optional): The scale tensor. Defaults to ``None``.
        zero (torch.Tensor, optional): The zero point tensor. Defaults to ``None``.

    Returns:
        tuple[QuantScale, torch.Tensor]: The scale and the zero point.
    """
    shape = tensor.shape
    # region step 1: prepare the linear scale and exponential scale dtypes, ranges, and view shapes
    scale_view_shapes = ShapeUtils.infer_scale_view_shapes(group_shapes=group_shapes, shape=shape)
    if scale is not None:
        scale = scale.view(scale_view_shapes[-1])
    if exponent_scaling_level is None:
        exponent_scaling_level = ScaleUtils.infer_exponent_scaling_level(scale_quant_dtypes)
    if exponent_scaling_level >= 0 and exponent_scaling_level < len(scale_quant_dtypes):
        lin_s_dtypes = scale_quant_dtypes[:exponent_scaling_level]
        exp_s_dtypes = scale_quant_dtypes[exponent_scaling_level:]
        lin_s_view_shapes = scale_view_shapes[:exponent_scaling_level]
        exp_s_view_shapes = scale_view_shapes[exponent_scaling_level:]
        exp_s_spans = infer_scale_quant_spans(exp_s_dtypes)
        if lin_s_dtypes:
            lin_s_spans = [lin_s_span * exp_s_spans[-1] for lin_s_span in infer_scale_quant_spans(lin_s_dtypes)]
        else:
            lin_s_spans = []
    else:
        lin_s_dtypes, exp_s_dtypes = scale_quant_dtypes, []
        lin_s_view_shapes, exp_s_view_shapes = scale_view_shapes, []
        lin_s_spans, exp_s_spans = infer_scale_quant_spans(lin_s_dtypes), []
    # endregion
    # region step 2: get linear quant span, exponential quant span
    quant_range = quant_range or QuantRange.build(quant_dtype)
    quant_range = quant_range.intersect(quant_dtype)
    if quant_dtype.has_zero_point:
        lin_quant_span = quant_range.max - quant_range.min
        exp_quant_span = 2 ** int(math.log2(quant_range.max) + int(quant_dtype.signed))
    else:
        lin_quant_span = quant_range.max
        exp_quant_span = 2 ** int(math.log2(quant_range.max))
    # endregion
    # region step 3: get the dynamic span for range-based scale or the scale tensor
    if scale is None:
        range_based = True
        if view_shape is None:
            view_shape = ShapeUtils.infer_view_shape(shape, group_shape=group_shapes[-1])
        dynamic_range = dynamic_range or DynamicRange()
        dynamic_range = dynamic_range.measure(
            tensor.view(view_shape), has_zero_point=quant_dtype.has_zero_point, is_float=quant_dtype.is_float
        )
        dynamic_span = (dynamic_range.max - dynamic_range.min) if quant_dtype.has_zero_point else dynamic_range.max
    else:
        range_based = False
        assert isinstance(scale, torch.Tensor), "Scale must be a tensor."
        scale = scale.to(dtype=tensor.dtype, device=tensor.device)
    # endregion
    # region step 4: get the scale
    if lin_s_dtypes:
        if range_based:
            lin_scale = dynamic_span / lin_quant_span
        elif exp_s_dtypes:
            lin_scale = scale.mul(exp_quant_span).div(lin_quant_span)
        else:
            lin_scale = scale
        lin_s = quantize_scale_kernel(
            lin_scale,
            quant_dtypes=lin_s_dtypes,
            quant_spans=lin_s_spans,
            view_shapes=lin_s_view_shapes,
        )
        assert lin_s.data is not None, "Linear scale tensor is None."
        assert not lin_s.data.isnan().any(), "Linear scale tensor contains NaN."
        assert not lin_s.data.isinf().any(), "Linear scale tensor contains Inf."
    else:
        lin_s = QuantScale()
    if exp_s_dtypes:
        if range_based:
            exp_scale = dynamic_span / exp_quant_span
        else:
            exp_scale = scale
        if lin_s.data is not None:
            lin_s.data = lin_s.data.expand(lin_s_view_shapes[-1]).reshape(scale_view_shapes[-1])
            exp_scale = exp_scale / lin_s.data
        exp_s = quantize_scale_kernel(
            exp_scale,
            quant_dtypes=exp_s_dtypes,
            quant_spans=exp_s_spans,
            view_shapes=exp_s_view_shapes,
        )
        assert exp_s.data is not None, "Exponential scale tensor is None."
        assert not exp_s.data.isnan().any(), "Exponential scale tensor contains NaN."
        assert not exp_s.data.isinf().any(), "Exponential scale tensor contains Inf."
        s = exp_s if lin_s.data is None else lin_s.extend(exp_s)
    else:
        s = lin_s
    assert s.data is not None, "Scale tensor is None."
    assert not s.data.isnan().any(), "Scale tensor contains NaN."
    assert not s.data.isinf().any(), "Scale tensor contains Inf."
    # endregion
    # region step 5: get the zero point
    if quant_dtype.has_zero_point:
        if range_based:
            zero = quant_range.min - dynamic_range.min / s.data
        else:
            assert isinstance(zero, torch.Tensor), "Zero point must be a tensor."
        # change the zero point dtype with the scale dtype
        z = simple_quantize(zero, quant_dtype=lin_s_dtypes[-1] if lin_s_dtypes else exp_s_dtypes[-1])
    else:
        z = torch.tensor(0, dtype=s.data.dtype, device=s.data.device)
    assert not z.isnan().any(), "Zero point tensor contains NaN."
    assert not z.isinf().any(), "Zero point tensor contains Inf."
    # endregion
    return s, z


def calibrate_scale_and_zero(
    tensor: torch.Tensor,
    tensor_hat: torch.Tensor,
    *,
    config: QuantGPTQConfig | QuantDecoupleQConfig,
    view_shape: torch.Size,
    scale: torch.Tensor,
    zero: torch.Tensor,
    inputs: ActivationsCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calibrate the scale and zero point of the tensor to be quantized."""
    """ Use PDE of loss function || WA - /hat{W}A ||_2 ^2 to calibrate """

    # region step 1: reshape the tensor and scale
    # view shape = (channel, 1, num_group 0, group_size 0, num_group 1, group_size 1)
    view_tensor = tensor.view(view_shape)
    len_view_shape = len(view_shape)
    reshaped_tensor = view_tensor.permute(0, 1, *range(2, len_view_shape, 2), *range(3, len_view_shape, 2))
    reshaped_tensor_hat = tensor_hat.view(view_shape).permute(0, 1, *range(2, len_view_shape, 2), *range(3, len_view_shape, 2))
    # reshaped_tensor: (#g0 * gs0, #g1 * #g2 * ... * gs1 * gs2 * ...)
    reshaped_tensor = reshaped_tensor.reshape(view_shape[0] * view_shape[1], -1)
    reshaped_tensor_hat = reshaped_tensor_hat.reshape(view_shape[0] * view_shape[1], -1)
    num_row_groups, num_column_groups = view_shape[0], view_shape[2::2].numel()
    row_group_size, column_group_size = view_shape[1], view_shape[3::2].numel()
    num_rows, num_columns = reshaped_tensor.shape
    reshaped_scale = scale.view(num_row_groups, 1, num_column_groups)
    zero_is_number = isinstance(zero, (int, float)) or zero.numel() == 1
    reshaped_zero = zero if zero_is_number else zero.view(num_row_groups, 1, num_column_groups)
    # endregion

    # region step 2: get the Hessian Matrix
    assert inputs.num_sources == 1, f"GPTQ requires only one input source, got {inputs.num_sources}."
    num_samples = inputs.num_samples
    xs, dim, fn = inputs[0].cached, inputs[0].channels_dim, inputs[0].transform
    hessian = torch.zeros((num_columns, num_columns), device=view_tensor.device, dtype=view_tensor.dtype)
    for x in xs:
        x: torch.Tensor = fn(x.view(-1, *x.shape[dim:]))
        if config.hessian_block_size > 0 and x.shape[0] > config.hessian_block_size:
            for b in range(0, x.shape[0], config.hessian_block_size):
                _x = x[b : min(b + config.hessian_block_size, x.shape[0])]
                _x = math.sqrt(2 / num_samples) * _x.to(device=view_tensor.device, dtype=view_tensor.dtype)
                hessian += torch.matmul(_x.t(), _x)
        else:
            x = math.sqrt(2 / num_samples) * x.to(device=view_tensor.device, dtype=view_tensor.dtype)
            hessian += torch.matmul(x.t(), x)
    dead = hessian.diagonal() == 0
    hessian[dead, dead] = 1
    reshaped_tensor[:, dead] = 0
    del xs, dim, fn, x, inputs, num_samples, dead
    gc.collect()
    torch.cuda.empty_cache()
    # endregion

    # region step 3: apply dampening to avoid numerical instability
    hessian_diag = hessian.diagonal()
    hessian_diag_mean = hessian_diag.mean()
    hessian_diag += config.damp_percentage * hessian_diag_mean
    del hessian_diag, hessian_diag_mean
    # endregion
    
    #
