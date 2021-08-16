#!/usr/bin/env python
# coding: utf-8
"""
Utility functions for MCP3 data reduction
"""
from typing import Tuple

import numpy as np


def empty_image(
    nrow: int = 512,
    ncol: int = 512,
    step: float = 0.25,
) -> Tuple[np.ndarray]:
    """
    Return an empty numpy array representation of the image

    @param nrow: number of rows on MCP3 detector sensor
    @param ncol: number of cols on MCP3 detector sensor
    @param step: substeps to provide super resolution
    """
    xx, yy = np.meshgrid(np.arange(0, nrow, step), np.arange(0, ncol, step))
    return xx, yy, np.zeros_like(xx, dtype=float)


if __name__ == "__main__":
    print("Utility modules for MCP3 data reduction package")
