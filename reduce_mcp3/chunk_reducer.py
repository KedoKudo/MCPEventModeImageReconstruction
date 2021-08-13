#!/usr/bin/env python
# coding: utf-8
"""
Reduce a single chunk of events.

Usage:
    chunk_reducer INPUTH5 STARTAT CHUNKSIZE DOMAINSIZE
    chunk_reducer -h | --help

Options:
    -h --help     Show this screen.
"""

import os
import logging
import h5py
import numpy as np
import concurrent.futures as cf
from multiprocessing import cpu_count
from docopt import docopt
from tqdm import tqdm

from .io import events_from_h5
from .clustering import cluster_domain_DBScan
# from .weighted_centroid import events_to_img
# from .gaussian_fit import events_to_img
from .gaussian_fit_fast import events_to_img
from .util import empty_image


def reduce_chunck_to_img(
    h5filename: str,
    startrow: int,
    chunksize: int,
    domainsize: int,
    mcp_nrow: int = 512,
    mcp_ncol: int = 512,
    subpixel: float = 0.125,
    minimum_events_per_cluster: int = 30,
) -> None:
    """reduce the given chunk into an image and save it to an HDF5"""
    #
    # setup logger
    #
    fnbase = os.path.basename(h5filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{fnbase}.log"),
            logging.StreamHandler()
        ],
    )
    #
    # read in events
    #
    events = events_from_h5(h5filename, startrow, chunksize)
    logging.info(f"Reading in chunk of events:\n{events}")
    #
    # use concurrent.futures to process chunk with overlapping domain
    #
    domain_starts = range(0, chunksize, int(0.95 * domainsize))
    ncpu = max(1, cpu_count() - 1)
    logging.info(f"Plan and execute clustering of domains")
    with cf.ProcessPoolExecutor(ncpu) as e:
        with tqdm(total=len(domain_starts)) as progress:
            # plan
            futures = []
            for st in domain_starts:
                endpos = min(st + domainsize, events.n_events)
                x = events.x_asfloat[st:endpos]
                y = events.y_asfloat[st:endpos]
                t = events.time[st:endpos]
                future = e.submit(cluster_domain_DBScan, x, y, t)
                future.add_done_callback(lambda _: progress.update())
                futures.append(future)
            # execute
            results = [future.result() for future in futures]
    #
    # update cid
    #
    # NOTE: due to the overlapping among domains, this has to be serial
    logging.info("Assign and pruning cluster IDs.")
    for cid, st in zip(results, domain_starts):
        if cid is None:
            continue
        endpos = min(st + domainsize, events.n_events)
        cid[cid > -1] += events.cluster_ids.max() + 1
        events.cluster_ids[st:endpos] = cid
    #
    # remove noises
    #
    logging.info("Remove non-cluster pts.")
    idx = np.where(events.cluster_ids > -0.5)[0]
    cid = events.cluster_ids[idx]
    x = events.x_asfloat[idx]
    y = events.y_asfloat[idx]
    tot = events.tot_asfloat[idx]
    #
    # cluster to img
    #
    logging.info(f"--{np.unique(cid).shape[0]} clusters to image with numba.")
    x_img, y_img, _ = empty_image(nrow=mcp_nrow, ncol=mcp_ncol, step=subpixel)
    img = events_to_img(
        x,  # event x position, must be float
        y,  # event y position, must be float
        tot,  # event time of threshold (as intensity), must be float
        cid,  # ID labels
        x_img,  # pixel position of the target image (see np.meshgrid)
        y_img,  # pixel position of the target image (see np.meshgrid)
        minimum_events_per_cluster,  # minimum number of events needed to be included
    )
    #
    # appending the image to HDF archive
    #
    oh5fn = os.path.basename(h5filename).replace(".h5", "_reduced.h5")
    chunkid = int(startrow / chunksize)
    logging.info(f"Append image to {oh5fn}")
    with h5py.File(oh5fn, "a") as oh5f:
        dset = oh5f.create_dataset(
            f"Chunk_{chunkid}",
            data=img,
            compression="gzip",
            compression_opts=7,
        )
        # add processing meta data
        dset.attrs["events_summary"] = str(events)
        dset.attrs["domainsize"] = domainsize
        dset.attrs["startrow"] = startrow
        dset.attrs["(mcp_nrow, mcp_ncol)"] = (mcp_nrow, mcp_ncol)
        dset.attrs["minimum_events_per_cluster"] = minimum_events_per_cluster


if __name__ == "__main__":
    arguments = docopt(__doc__)
    reduce_chunck_to_img(
        arguments["INPUTH5"],
        int(arguments["STARTAT"]),
        int(arguments["CHUNKSIZE"]),
        int(arguments["DOMAINSIZE"]),
    )
