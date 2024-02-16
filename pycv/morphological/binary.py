import numpy as np
from pycv._lib._src_py import pycv_morphology

__all__ = [
    'binary_erosion',
    'binary_dilation',
    'binary_opening',
    'binary_closing',
    'binary_edge',
    'skeletonize',
    'remove_small_objects',
    'remove_small_holes',
    'binary_hit_or_miss',
    'binary_fill_holes'
]


########################################################################################################################


def binary_erosion(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        iterations: int = 1,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
        border_val: int = 0,
        extra_memory: bool = True
) -> np.ndarray:
    ret = pycv_morphology.binary_erosion(image, strel, offset, iterations, mask, output, 0, border_val, extra_memory)
    return output if ret is None else ret


def binary_dilation(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        iterations: int = 1,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
        border_val: int = 0,
        extra_memory: bool = True
) -> np.ndarray:
    ret = pycv_morphology.binary_erosion(image, strel, offset, iterations, mask, output, 1, border_val, extra_memory)
    return output if ret is None else ret


########################################################################################################################

def binary_opening(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
        border_val: int = 0
) -> np.ndarray:
    ero = pycv_morphology.binary_erosion(image, strel, offset, 1, mask, None, 0, border_val)
    ret = pycv_morphology.binary_erosion(ero, strel, offset, 1, mask, output, 1, border_val)
    return output if ret is None else ret


def binary_closing(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
        border_val: int = 0
) -> np.ndarray:
    dil = pycv_morphology.binary_erosion(image, strel, offset, 1, mask, None, 1, border_val)
    ret = pycv_morphology.binary_erosion(dil, strel, offset, 1, mask, output, 0, border_val)
    return output if ret is None else ret


def binary_edge(
        image: np.ndarray,
        edge_mode: str = 'inner',
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
        border_val: int = 0
) -> np.ndarray:
    supported_mode = {'inner', 'outer', 'double'}
    if edge_mode not in supported_mode:
        raise ValueError(f'{edge_mode} mode not supported use one of: {supported_mode}')

    dil = None
    ero = None

    if edge_mode != 'inner':
        dil = pycv_morphology.binary_erosion(image, strel, offset, 1, mask, None, 1, border_val)
    if edge_mode != 'outer':
        ero = pycv_morphology.binary_erosion(image, strel, offset, 1, mask, None, 0, border_val)

    if output is None:
        output = np.zeros_like(dil if dil is not None else ero)

    if ero is not None and dil is not None:
        output[:] = dil ^ ero
    elif dil is not None:
        output[:] = dil ^ image
    else:
        output[:] = ero ^ image
    return output


########################################################################################################################

def skeletonize(
        image: np.ndarray
) -> np.ndarray:
    return pycv_morphology.skeletonize(image)


########################################################################################################################

def remove_small_objects(
        image: np.ndarray,
        threshold: int = 32,
        connectivity: int = 1
) -> np.ndarray:
    return pycv_morphology.remove_small_objects(image, threshold, connectivity)


def remove_small_holes(
        image: np.ndarray,
        threshold: int = 32,
        connectivity: int = 1
) -> np.ndarray:
    return pycv_morphology.remove_small_objects(image, threshold, connectivity, invert=1)


########################################################################################################################

def binary_hit_or_miss(
        image: np.ndarray,
        strel1: np.ndarray | None = None,
        strel2: np.ndarray | None = None,
        offset1: tuple | None = None,
        offset2: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
        border_val: int = 0
) -> np.ndarray:
    ret = pycv_morphology.binary_hit_or_miss(image, strel1, strel2, offset1, offset2, mask, output, border_val)
    return output if ret is None else ret


########################################################################################################################

def binary_fill_holes(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        output: np.ndarray | None = None,
        extra_memory: bool = True
) -> np.ndarray:
    inputs = np.zeros_like(image)
    inputs_mask = image == 0

    ret = pycv_morphology.binary_erosion(inputs, strel, offset, -1, inputs_mask, output, 1, 1, extra_memory)
    if ret is None:
        out = output
    else:
        out = ret
    np.logical_not(out, out)
    return out

########################################################################################################################
