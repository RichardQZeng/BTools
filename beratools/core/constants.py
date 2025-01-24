import numpy as np
import enum 

NADDatum = ['NAD83 Canadian Spatial Reference System', 'North American Datum 1983']

ASSETS_PATH = "assets"
BT_DEBUGGING = False
BT_SHOW_ADVANCED_OPTIONS = False
BT_UID = 'BT_UID'

BT_EPSILON = np.finfo(float).eps
BT_NODATA_COST = np.inf
BT_NODATA = -9999

# To be removed
LP_SEGMENT_LENGTH = 500
FP_CORRIDOR_THRESHOLD = 2.5

# centerline
CL_USE_SKIMAGE_GRAPH = False
CL_DELETE_HOLES = True
CL_SIMPLIFY_POLYGON = True

class FootprintParams(float, enum.Enum):
    FP_CORRIDOR_THRESHOLD = 2.5

class CenterlineParams(float, enum.Enum):
    BUFFER_CLIP = 5.0
    SEGMENTIZE_LENGTH = 1.0
    SIMPLIFY_LENGTH = 0.5
    SMOOTH_SIGMA = 0.8
    CLEANUP_POLYGON_BY_AREA = 1.0
    POLYGON_BUFFER = 1

class CenterlineFlags(enum.Flag):
    USE_SKIMAGE_GRAPH = False
    DELETE_HOLES = True
    SIMPLIFY_POLYGON = True

@enum.unique
class CenterlineStatus(enum.IntEnum):
    SUCCESS = 1
    FAILED = 2
    REGENERATE_SUCCESS = 3
    REGENERATE_FAILED = 4

@enum.unique
class ParallelMode(enum.IntEnum):
    SEQUENTIAL = 1
    MULTIPROCESSING = 2
    CONCURRENT = 3
    DASK = 4
    SLURM = 5
    # RAY = 6

PARALLEL_MODE = ParallelMode.MULTIPROCESSING
