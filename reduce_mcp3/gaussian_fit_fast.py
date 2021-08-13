#!/usr/bin/env python
# coding: utf-8
"""
This module is converting a sanitized and clustered events into an image
using gaussian fit method.
The selected gaussian fit method is an appoximation of the gaussian fit using least squares.
For more infomration, refers to DOI: 10.1021/la900393v

The input event should have integer as cluster ID.
"""

import numpy as np
import numba


@numba.jit(nopython=True, parallel=True)
def events_to_img(
        xs: np.ndarray,
        ys: np.ndarray,
        tots: np.ndarray,
        cluster_ids: np.ndarray,
        x_img: np.ndarray,
        y_img: np.ndarray,
        minimum_event_num: int = 30,
        extinguish_dist: float = 1.41422,  # sqrt(2) = 1.41421356237
) -> np.ndarray:
    """
    Converting given events into a flatten image array defined by
    given image pixel positions

    @param xs: event x position, must be float
    @param ys: event y position, must be float
    @param tots: event time of threshold (as intensity), must be float
    @param cluster_ids: ID labels
    @param x_img: pixel position of the target image (see np.meshgrid)
    @param y_img: pixel position of the target image (see np.meshgrid)
    @param minimum_event_num: minimum number of events needed to be included
    @param extinguish_dist: signal impact ends outside this range

    @returns: the image converted from given events using least-square approximated gaussian fitting
    """
    # preparation
    unique_cids = np.unique(cluster_ids)
    img_shape = x_img.shape
    x_img = x_img.flatten()
    y_img = y_img.flatten()
    img = x_img * 0.0
    for i in numba.prange(unique_cids.shape[0]):
        cid = unique_cids[i]
        idx = np.where(cluster_ids == cid)[0]
        # skip cluster with too few events
        if idx.shape[0] < minimum_event_num:
            continue
        # compute the centroid position and weighted equivalent intensity
        # - get the center from least square fitting
        # - replace the extintion distance with two sigmas
        # - use the amplitude of the gaussian fit as the intensity
        #   - NOTE: we are still not sure whether we should consider the amplitudes
        #           or only use the single event count here
        intensity = tots[idx] * 1.0  # a copy of the intensity
        xdata = xs[idx]
        ydata = ys[idx]
        background = np.median(tots[idx])
        intensity = tots[idx] - background
        mask = np.where(intensity > 0)[0]
        weight = intensity[mask]**2
        # - solve for Ax = b
        b = (xdata[mask]**2 + ydata[mask]**2) * weight
        A = (np.column_stack((xdata[mask], ydata[mask], np.log(
            intensity[mask]), np.ones(weight.size))).T * weight).T
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=-1.0)
        a1, a2, a3, _ = x
        xc, yc, _ = a1 / 2, a2 / 2, np.sqrt(a3 / -2)
        # propogate the signal to the image
        idx = np.where(
            np.logical_and(
                np.logical_and(
                    x_img >= xc - extinguish_dist,
                    x_img < xc + extinguish_dist,
                ),
                np.logical_and(
                    y_img >= yc - extinguish_dist,
                    y_img < yc + extinguish_dist,
                ),
            ))[0]
        wgts = (x_img[idx] - xc)**2 + (y_img[idx] - yc)**2
        wgts = 1.0 / wgts
        wgts = wgts / np.sum(wgts)
        img[idx] += wgts
    # return the results
    return img.reshape(img_shape)


if __name__ == "__main__":
    print(__doc__)