#!/usr/bin/env python
# coding: utf-8
"""
Use scikit-learn's DBScan to cluster a domain of events.

NOTE:
Using the testing data, a domain of 320_000 events has a rough
time span of 7ms.
Generally speaking, 100 domains forms a chunk, there for one chunk
should contains 32_000_000 events with a rough time span of 1s.
It should be noted that events are not uniformlly distributed along
the time axis.
"""
import numpy as np
from sklearn.cluster import DBSCAN


def cluster_domain_DBScan(
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    """
    return the cluster ID based on the given data using DBScan

    @param x: event position x vector
    @param y: event position y vector
    @param time: event TOA vector

    @returns: clustering results
    """
    # 25ns is the step clock, 40 MHz
    features = np.column_stack((x, y, time / 25))
    try:
        clustering = DBSCAN(
            eps=2 * np.sqrt(3),  # search vol, sphere of unit two
            min_samples=4,  # 4 neighbors are required to qualify as a cluster
            leaf_size=400_000,  # large leaf size for faster clustering
            n_jobs=1,  # parallel on the top level
        ).fit(features)
        return clustering.labels_
    except RuntimeError:
        return None


if __name__ == "__main__":
    pass
