#!/usr/bin/env python
# coding: utf-8
"""
Top level control to reduce raw data from MCP3 detector to
images.

Usage:
    reducer INPUTH5
    reducer INPUTH5 CHUNKSIZE
    reducer INPUTH5 CHUNKSIZE DOMAINSIZE
    reducer INPUTH5 --restart <startat>
    reducer -h | --help

Options:
    -h --help     Show this screen.
    -r --restart  Restart at given row.
"""
import logging
import os
import subprocess

import h5py
from docopt import docopt

# NOTE:
# The event here refers to the excitation of timpix3 sensor, which is different
# from event based neutron characterization technique.

# USEFUL CONSTANTS
DOMAINSIZE = 32_000  # equivalent to about 1 ms in experiment time
CHUNKSIZE = 32_000_000  # equivalent to about 1 sec in experiment time


def get_total_events(h5fn: str) -> int:
    """Return the total number of events"""
    with h5py.File(h5fn, "r") as h5f:
        n_events = h5f["x"].shape[0]
    return n_events


if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)

    #
    # parse input arguments
    #
    h5fn = arguments["INPUTH5"]
    #
    if arguments["--restart"]:
        startat = int(arguments["<startat>"])
    else:
        startat = 0
    #
    if arguments["CHUNKSIZE"] is None:
        chunksize = CHUNKSIZE
    else:
        chunksize = int(arguments["CHUNKSIZE"])
    #
    if arguments["DOMAINSIZE"] is None:
        domainsize = DOMAINSIZE
    else:
        domainsize = int(arguments["DOMAINSIZE"])
    #
    # setup logger
    #
    fnbase = os.path.basename(h5fn)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{fnbase}.log"),
            logging.StreamHandler()
        ],
    )
    #
    logging.info(f"Start reducing {h5fn} to image.")
    #
    total_events = get_total_events(h5fn)
    # prepare sub-processing calls
    for _, start_row in enumerate(range(startat, total_events, chunksize)):
        cmd = [
            "python", "-m", "reduce_mcp3.chunk_reducer", h5fn,
            str(start_row),
            str(chunksize),
            str(domainsize)
        ]
        logging.info(" ".join(cmd))
        p = subprocess.Popen(cmd)
        p.wait()
