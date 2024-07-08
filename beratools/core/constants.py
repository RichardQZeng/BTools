import numpy as np
from enum import Flag, Enum, IntEnum, unique


NADDatum = ['NAD83 Canadian Spatial Reference System', 'North American Datum 1983']

BT_EPSILON = np.finfo(float).eps
BT_NODATA_COST = np.inf
BT_NODATA = -9999
BT_DEBUGGING = False
BT_MAXIMUM_CPU_CORES = 60  # multiprocessing has limit of 64, consider pathos
BT_BUFFER_RATIO = 0.0  # overlapping ratio of raster when clipping lines
BT_LABEL_MIN_WIDTH = 130
BT_SHOW_ADVANCED_OPTIONS = False
BT_UID = 'BT_UID'

GROUPING_SEGMENT = True
LP_SEGMENT_LENGTH = 500

FP_CORRIDOR_THRESHOLD = 2.5
FP_SEGMENTIZE_LENGTH = 2.0
FP_FIXED_WIDTH_DEFAULT = 5.0
FP_PERP_LINE_OFFSET = 30.0

# centerline
CL_USE_SKIMAGE_GRAPH = False
CL_BUFFER_CLIP = 5.0
CL_BUFFER_CENTROID = 3.0
CL_SNAP_TOLERANCE = 15.0
CL_SEGMENTIZE_LENGTH = 1.0
CL_SIMPLIFY_LENGTH = 0.5
CL_SMOOTH_SIGMA = 0.8
CL_DELETE_HOLES = True
CL_SIMPLIFY_POLYGON = True
CL_CLEANUP_POLYGON_BY_AREA = 1.0
CL_POLYGON_BUFFER = 1e-6


class FloatEnum(float, Enum):
    VERY_DRY = 0.2
    DRY = 0.5
    MEDIAN = 0.8
    WET = 1.0


class Boolean(Flag):
    a = True
    b = False


@unique
class CenterlineStatus(IntEnum):
    SUCCESS = 1
    FAILED = 2
    REGENERATE_SUCCESS = 3
    REGENERATE_FAILED = 4


@unique
class ParallelMode(IntEnum):
    SEQUENTIAL = 1
    MULTIPROCESSING = 2
    DASK = 3
    RAY = 4


PARALLEL_MODE = ParallelMode.MULTIPROCESSING


