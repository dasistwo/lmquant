# -*- coding: utf-8 -*-
"""Various functions for GPTQ and TrueNibble Quantization."""

import torch
import logging
import math
import gc
from typing import Tuple

from ...dataset import ActivationsCache
from .config import QuantGPTQConfig, QuantDecoupleQConfig

__all__ = ["generate_hessian", "cal_quant_error"]


def generate_hessian(
    inputs: ActivationsCache,
    weight: torch.Tensor,
    config: QuantGPTQConfig | QuantDecoupleQConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # region step 1: get Hessian matrix
    assert inputs.num_sources == 1, f"generate_hessian requires only one input source, got {inputs.num_sources}."
    _, num_columns = weight.shape
    num_samples = inputs.num_samples
    xs, dim, fn = inputs[0].cached, inputs[0].channels_dim, inputs[0].transform
    hessian = torch.zeros((num_columns, num_columns), device=weight.device, dtype=weight.dtype)
    for x in xs:
        x: torch.Tensor = fn(x.view(-1, *x.shape[dim:]))
        if config.hessian_block_size > 0 and x.shape[0] > config.hessian_block_size:
            for b in range(0, x.shape[0], config.hessian_block_size):
                _x = x[b : min(b + config.hessian_block_size, x.shape[0])]
                _x = math.sqrt(2 / num_samples) * _x.to(device=weight.device, dtype=weight.dtype)
                hessian += torch.matmul(_x.t(), _x)
        else:
            x = math.sqrt(2 / num_samples) * x.to(device=weight.device, dtype=weight.dtype)
            hessian += torch.matmul(x.t(), x)
    dead = hessian.diagonal() == 0
    hessian[dead, dead] = 1
    weight[:, dead] = 0
    del xs, dim, fn, x, inputs, num_samples, dead
    gc.collect()
    torch.cuda.empty_cache()
    # endregion
    # region step 2: permute the Hessian matrix : actorder
    importance = torch.diag(hessian)  # (#g1 * #g2 * ... * gs1 * gs2 * ..., )
    permute = torch.argsort(importance, descending=True)
    hessian = hessian[permute][:, permute]
    del importance
    # endregion
    # region step 3: apply dampening to avoid numerical instability
    hessian_diag = hessian.diagonal()
    hessian_diag_mean = hessian_diag.mean()
    hessian_diag += config.damp_percentage * hessian_diag_mean
    # endregion
    # region step 4: get the inverse of the Hessian matrix
    stable_inv, num_inv_tries = False, 0
    while (not stable_inv) and num_inv_tries < config.num_inv_tries:
        num_inv_tries += 1
        try:
            hessian_inv = torch.linalg.cholesky(hessian)
            hessian_inv = torch.cholesky_inverse(hessian_inv)
            hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)
        except RuntimeError:
            hessian_diag += (config.damp_percentage * 0.1) * hessian_diag_mean
            continue
        stable_inv = True
    if num_inv_tries > 1:
        logger = logging.getLogger(f"{__name__}.GPTQ")
        logger.debug(
            "        - GPTQ Hessian is not stable %s %d tries.", "until" if stable_inv else "after", num_inv_tries
        )
    assert not hessian_inv.isinf().any(), "Inverse of Hessian matrix contains Inf."
    assert not hessian_inv.isnan().any(), "Inverse of Hessian matrix contains NaN."
    del (
        hessian_diag,
        hessian_diag_mean,
        num_inv_tries,
    )
    return (
        hessian,
        hessian_inv,
        permute,
    )
    # endregion


def cal_quant_error(x, q, h):
    assert x.ndim == 2
    assert q.ndim == 2
    assert h.ndim == 2
    err = torch.matmul(
        torch.matmul(q - x, h), (q - x).t()
    ).diag()  # This line is much faster than the three lines below
    # r = q - x
    # err = torch.matmul(torch.matmul(r.unsqueeze(1), h.unsqueeze(0)), r.unsqueeze(-1))
    # err = err.squeeze(-1).squeeze(-1)
    return err  # [num_channel]
