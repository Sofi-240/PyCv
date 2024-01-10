import numpy as np
import math
import numbers
from pycv._lib.filters_support.kernel_utils import reshape_1d_kernel, default_binary_strel
from pycv._lib.array_api.dtypes import cast

__all__ = [
    'binary_strel',
    'rectangle',
    'square',
    'cube',
    'cuboid',
    'cross',
    'diamond',
    'octahedron',
    'disk',
    'circle',
    'sphere',
    'cylinder',
    'SOBEL_EDGE',
    'SOBEL_WEIGHTS',
    'PREWITT_EDGE',
    'PREWITT_WEIGHTS',
    'edge_kernel',
    'gaussian_kernel',
    'PUBLIC'
]
PUBLIC = []


########################################################################################################################
def _is_odd(_int):
    return _int % 2 != 0


########################################################################################################################
# binary
def _disk_shape(radius: int, ndim: int = 2, extend: bool = True):
    axs = np.meshgrid(*(np.arange(-radius, radius + 1),) * ndim)
    grid = axs[0] ** 2
    for a in axs[1:]:
        grid += (a ** 2)
    if extend:
        radius += 0.5
    return np.array(grid <= (radius ** 2), dtype=bool)


def binary_strel(
        ndim: int,
        connectivity: int = 1,
        hole: bool = False
) -> np.ndarray:
    return default_binary_strel(ndim, connectivity, hole)


def rectangle(height: int, width: int) -> np.ndarray:
    return np.ones((height, width), dtype=bool)


def square(width: int) -> np.ndarray:
    return np.ones((width,) * 2, dtype=bool)


def cube(width: int) -> np.ndarray:
    return np.ones((width,) * 3, dtype=bool)


def cuboid(height: int, width: int, length: int) -> np.ndarray:
    return np.ones((height, width, length), dtype=bool)


def cross(shape: tuple) -> np.ndarray:
    ndim = len(shape)
    if ndim < 2 or ndim > 3:
        raise ValueError(f'supported number of dimensions is 2D or 3D got {ndim}')
    se = np.zeros(shape, dtype=bool)
    mid = tuple(s // 2 if _is_odd(s) else slice(s // 2 - 1, s // 2 + 1) for s in shape)
    for i in range(len(mid)):
        se[mid[:i] + (slice(None),) + mid[i + 1:]] = 1
    return se


def diamond(radius: int) -> np.ndarray:
    y, x = np.meshgrid(*(np.arange(0, radius * 2 + 1),) * 2)
    return np.array(np.abs(y - radius) + np.abs(x - radius) <= radius, dtype=bool)


def octahedron(radius: int) -> np.ndarray:
    y, x, z = np.meshgrid(*(np.arange(0, radius * 2 + 1),) * 3)
    return np.array(np.abs(y - radius) + np.abs(x - radius) + np.abs(z - radius) <= radius, dtype=bool)


def disk(radius: int) -> np.ndarray:
    return _disk_shape(radius, ndim=2, extend=False)


def circle(radius: int) -> np.ndarray:
    return _disk_shape(radius, ndim=2, extend=True)


def sphere(radius: int) -> np.ndarray:
    return _disk_shape(radius, ndim=3, extend=True)


def cylinder(radius: int, height: int) -> np.ndarray:
    base = circle(radius)
    out = np.zeros((height, *base.shape), dtype=bool)
    out[:] = base
    return out


########################################################################################################################
# Edge Kernels


SOBEL_EDGE = np.array([1, 0, -1], dtype=np.float32)
SOBEL_WEIGHTS = np.array([1, 2, 1], dtype=np.float32) / 4

PREWITT_EDGE = np.array([1, 0, -1], dtype=np.float32)
PREWITT_WEIGHTS = np.array([1, 1, 1], dtype=np.float32) / 3


def edge_kernel(
        smooth_values: np.ndarray,
        edge_values: np.ndarray,
        ndim: int,
        axis: int
) -> np.ndarray:
    kernel = reshape_1d_kernel(edge_values, ndim, axis)
    for filter_dim in filter(lambda a: a != axis, range(ndim)):
        kernel = kernel * reshape_1d_kernel(smooth_values, ndim, filter_dim)
    return kernel


########################################################################################################################
# Gaussian

def gaussian_kernel(
        sigma: float,
        ndim: int = 1,
        truncate: float = 3.,
        radius: tuple | int | None = None,
) -> np.ndarray:
    if radius is None:
        radius = math.floor(truncate * sigma + 0.5)
    if isinstance(radius, numbers.Number):
        radius = (int(radius),) * ndim
    ax = np.ogrid[[slice(-r, r + 1) for r in radius]]
    h = np.exp((-sum(a ** 2 for a in ax) / (2. * sigma ** 2)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sum_ = h.sum()
    if sum_ != 0: h /= sum_
    return cast(h, np.float32)

########################################################################################################################
