import numpy as np
from pycv.morphological._utils import binary_dispatcher
from pycv._lib.core import ops

__all__ = [
    'binary_erosion',
    'binary_dilation',
    'binary_opening',
    'binary_closing',
    'binary_edge',
    'PUBLIC'
]

PUBLIC = [
    'binary_erosion',
    'binary_dilation',
    'binary_opening',
    'binary_closing',
    'binary_edge',
]


########################################################################################################################

@binary_dispatcher
def binary_erosion(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        iterations: int = 1,  # TODO: iterations
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    ops.binary_erosion(image, strel, output, offset, iterations, mask, 0)

    return output


@binary_dispatcher
def binary_dilation(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        iterations: int = 1,  # TODO: iterations
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    ops.binary_erosion(image, strel, output, offset, iterations, mask, 1)
    return output


########################################################################################################################
@binary_dispatcher
def binary_opening(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    ero = np.zeros_like(output)
    ops.binary_erosion(image, strel, ero, offset, 1, mask, 0)
    ops.binary_erosion(image, strel, output, offset, 1, mask, 1)
    return output


@binary_dispatcher
def binary_closing(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    dil = np.zeros_like(output)
    ops.binary_erosion(image, strel, dil, offset, 1, mask, 1)
    ops.binary_erosion(image, strel, output, offset, 1, mask, 0)
    return output


@binary_dispatcher
def binary_edge(
        image: np.ndarray,
        edge_mode: str = 'inner',
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    supported_mode = {'inner', 'outer', 'double'}
    if edge_mode not in supported_mode:
        raise ValueError(f'{edge_mode} mode not supported use one of: {supported_mode}')

    dil = None
    ero = None

    if supported_mode != 'inner':
        dil = np.zeros_like(output)
        ops.binary_erosion(image, strel, dil, offset, 1, mask, 1)
    if supported_mode != 'outer':
        ero = np.zeros_like(output)
        ops.binary_erosion(image, strel, ero, offset, 1, mask, 0)

    if ero is not None and dil is not None:
        output[:] = dil ^ ero
    elif dil is not None:
        output[:] = dil ^ image
    else:
        output[:] = ero ^ image
    return output
