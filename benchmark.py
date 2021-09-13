#!/usr/bin/env python
# coding: utf-8
"""
Benchmark the reduction code using line_profiler

NOTE:
    This is meant for developers to assess the performance of the reduction
    at each stage, and is not meant to be used in production.
"""
from reduce_mcp3.chunk_reducer import reduce_chunck_to_img


def call_reducer():
    """
    Call the reduction code
    """
    h5filename = "data/testing_small.h5"
    startrow = 0
    chunksize = 800_000
    domainsize = 100_000
    reduce_chunck_to_img(h5filename, startrow, chunksize, domainsize)


if __name__ == "__main__":
    print("Start bench marking")
    call_reducer()
