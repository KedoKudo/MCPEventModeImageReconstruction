#!/usr/bin/env python
# coding: utf-8
"""
This module is converting a sanitized and clustered events into an image
using weighted centroid method.

The input event should have integer as cluster ID.
"""

import numba
import numpy as np


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

    @returns: the image converted from given events using weighted centroid method
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
        wgts = tots[idx] / np.sum(tots[idx])
        xc = np.dot(wgts, xs[idx])
        yc = np.dot(wgts, ys[idx])
        # ic = np.dot(wgts, tots[idx])
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
    print("Example usage of weighted centroid method.")
