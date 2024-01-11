import numpy as np
from pycv._lib.filters_support.morphology import c_binary_erosion

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


def binary_erosion(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        iterations: int = 1,  # TODO: iterations
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    ret = c_binary_erosion(image, strel, offset, iterations, mask, output, 0)
    return output if ret is None else ret


def binary_dilation(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        iterations: int = 1,  # TODO: iterations
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    ret = c_binary_erosion(image, strel, offset, iterations, mask, output, 1)
    return output if ret is None else ret


########################################################################################################################

def binary_opening(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    ero = c_binary_erosion(image, strel, offset, 1, mask, None, 0)
    ret = c_binary_erosion(ero, strel, offset, 1, mask, output, 1)
    return output if ret is None else ret


def binary_closing(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    dil = c_binary_erosion(image, strel, offset, 1, mask, None, 1)
    ret = c_binary_erosion(dil, strel, offset, 1, mask, output, 0)
    return output if ret is None else ret



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
        dil = c_binary_erosion(image, strel, offset, 1, mask, None, 1)
    if supported_mode != 'outer':
        ero = c_binary_erosion(image, strel, offset, 1, mask, None, 0)

    if output is None:
        output = np.zeros_like(dil if dil is not None else ero)

    if ero is not None and dil is not None:
        output[:] = dil ^ ero
    elif dil is not None:
        output[:] = dil ^ image
    else:
        output[:] = ero ^ image
    return output
