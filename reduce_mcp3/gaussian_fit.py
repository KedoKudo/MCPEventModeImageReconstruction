#!/usr/bin/env python
# coding: utf-8
"""
This module is converting a sanitized and clustered events into an image
using gaussian fit method.

The input event should have integer as cluster ID.
"""
import concurrent.futures as cf
import logging
from multiprocessing import cpu_count
from typing import Tuple

import lmfit
import numba
import numpy as np
from tqdm import tqdm


def fit_cluster_center(
    tot: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """
    Use lmfit to perform quick Gaussian fitting to locate peak center

    @param tot: time-over-threshold
    @param x: x pos vector
    @param y: y pos vector

    @returns: (x_center, y_center)
    """
    model = lmfit.models.Gaussian2dModel()
    params = model.guess(tot, x, y)
    result = model.fit(tot, x=x, y=y, params=params)
    xc = result.best_values["centerx"]
    yc = result.best_values["centery"]
    return xc, yc


@numba.jit(nopython=True, parallel=True)
def center_to_img(
    cluster_centers: np.ndarray,
    x_img: np.ndarray,
    y_img: np.ndarray,
    extinguish_dist: float = 1.41422,
) -> np.ndarray:
    """
    Use same inverse distance method to reconstruct the image based on center
    location.

    @param cluster_centers: list of cluster centers
    @param x_img: pixel position of the target image (see np.meshgrid)
    @param y_img: pixel position of the target image (see np.meshgrid)
    @param extinguish_dist: signal impact ends outside this range

    @returns: the image
    """
    img_shape = x_img.shape
    x_img = x_img.flatten()
    y_img = y_img.flatten()
    img = x_img * 0.0

    for i in numba.prange(len(cluster_centers)):
        xc = cluster_centers[i, 0]
        yc = cluster_centers[i, 1]
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
        # inverse distance as before, but single count
        wgts = (x_img[idx] - xc)**2 + (y_img[idx] - yc)**2
        wgts = 1.0 / wgts
        wgts = wgts / np.sum(wgts)
        img[idx] += wgts
    # return the results
    return img.reshape(img_shape)


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
    #
    logging.info("calculate centers for each cluster")
    unique_cids = np.unique(cluster_ids)
    ncpu = max(1, cpu_count() - 1)
    # fit for the center
    logging.info(
        "-- the percentage shows how many clusters have enough pts to be valid."
    )
    with cf.ProcessPoolExecutor(ncpu) as e:
        with tqdm(total=len(unique_cids)) as progress:
            # plan
            futures = []
            for cid in unique_cids:
                idx = np.where(cluster_ids == cid)[0]
                if len(idx) < minimum_event_num:
                    continue
                future = e.submit(
                    fit_cluster_center,
                    tots[idx],
                    xs[idx],
                    ys[idx],
                )
                future.add_done_callback(lambda _: progress.update())
                futures.append(future)
            # execute
            results = [future.result() for future in futures]
    #
    logging.info("convert cluter centers to image")
    results = np.array(results, dtype=float)
    return center_to_img(
        cluster_centers=results,
        x_img=x_img,
        y_img=y_img,
        extinguish_dist=extinguish_dist,
    )


if __name__ == "__main__":
    print("Example usage of Gaussian fit method.")
