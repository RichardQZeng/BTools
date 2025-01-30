import enum

import numpy as np

NADDatum = ['NAD83 Canadian Spatial Reference System', 'North American Datum 1983']

ASSETS_PATH = "assets"
BT_DEBUGGING = False
BT_UID = 'BT_UID'

BT_EPSILON = np.finfo(float).eps
BT_NODATA_COST = np.inf
BT_NODATA = -9999

LP_SEGMENT_LENGTH = 500
FP_CORRIDOR_THRESHOLD = 2.5
SMALL_BUFFER = 1e-3

class CenterlineFlags(enum.Flag):
    """Flags for the centerline algorithm."""

    USE_SKIMAGE_GRAPH = False
    DELETE_HOLES = True
    SIMPLIFY_POLYGON = True

@enum.unique
class ParallelMode(enum.IntEnum):
    """Defines the parallel mode for the algorithms."""

    SEQUENTIAL = 1
    MULTIPROCESSING = 2
    CONCURRENT = 3
    DASK = 4
    SLURM = 5
    # RAY = 6

PARALLEL_MODE = ParallelMode.MULTIPROCESSING
