#!/usr/bin/env python
# coding: utf-8
"""
Provide several useful auxilary functions to facilitate the data reduction.
"""

import h5py
import numpy as np
from dataclasses import dataclass


@dataclass
class Events:
    """Chunk of events from MCP3 detector (sorted by time)"""
    x: np.ndarray  # event position x
    y: np.ndarray  # event position y
    time: np.ndarray  # event TOA (time of arrival)
    tot: np.ndarray  # event TOT (time over threshold)

    def __post_init__(self):
        """sanitize input and create cluster IDs holder"""
        # remove nan and inf
        idx = np.where(
            np.logical_and.reduce((
                np.isfinite(self.x),
                np.isfinite(self.y),
                np.isfinite(self.time),
                np.isfinite(self.tot),
            )))[0]
        self.x = self.x[idx]
        self.y = self.y[idx]
        self.time = self.time[idx]
        self.tot = self.tot[idx]
        # sort along time axis
        idx = np.argsort(self.time)
        self.x = self.x[idx]
        self.y = self.y[idx]
        self.time = self.time[idx]
        self.tot = self.tot[idx]
        # create the cluster_id holder
        # NOTE: -1 is used to mark noise by scikit-learn DBScan
        self.cluster_ids = np.full(self.x.shape, -1, dtype=int)

    def __str__(self) -> str:
        outstr = "Events summary:"
        outstr += f"\n\tN: {self.n_events}"
        outstr += f"\n\ttime span: {self.timespan:,} (ns)"
        outstr += f"\n\tx span: {self.xspan}/512"
        outstr += f"\n\ty span: {self.yspan}/512"
        return outstr

    @property
    def TOT(self):
        """alias to tot"""
        return self.tot

    @property
    def x_asfloat(self) -> np.ndarray:
        return self.x.astype(float)

    @property
    def y_asfloat(self) -> np.ndarray:
        return self.y.astype(float)

    @property
    def tot_asfloat(self) -> np.ndarray:
        return self.tot.astype(float)

    @property
    def n_events(self) -> int:
        """number of events in the chunk"""
        return len(self.x)

    @property
    def timespan(self) -> int:
        """return the time span of the events"""
        return self.time.max() - self.time.min()

    @property
    def xspan(self) -> int:
        return self.x.max() - self.x.min()

    @property
    def yspan(self) -> int:
        return self.y.max() - self.y.min()


def events_from_h5(
    h5filename: str,
    start_at: int = 0,
    nrows: int = None,
) -> Events:
    """
    Read in given amount of entries (rows, a.k.a. events) and convert to a
    dict of numpy array.
    """
    with h5py.File(h5filename, "r") as h5f:
        # calc end idx
        if nrows is None:
            end_at = h5f["x"].shape[0]
        else:
            end_at = min(start_at + nrows, h5f["x"].shape[0])
        # build the chunk
        events = Events(
            h5f["x"][start_at:end_at],
            h5f["y"][start_at:end_at],
            h5f["time"][start_at:end_at],
            h5f["TOT"][start_at:end_at],
        )
    return events


if __name__ == "__main__":
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print("One domain")
    h5f = f"{dir_path}/../data/imaging_small_compressed.h5"
    events = events_from_h5(h5f, 0, 320_000)
    print(events)
    print("One chunck")
    events = events_from_h5(h5f, 0, 32_000_000)
    print(events)
    print("All")
    events = events_from_h5(h5f, 0, None)
    print(events)
