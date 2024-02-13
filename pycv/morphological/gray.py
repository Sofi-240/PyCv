import numpy as np
from pycv._lib._src_py import pycv_morphology

__all__ = [
    'gray_erosion',
    'gray_dilation',
    'gray_opening',
    'gray_closing',
    'black_top',
    'white_top',
    'area_open',
    'area_close'
]

########################################################################################################################

def gray_erosion(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
        border_val: int = 0
) -> np.ndarray:
    ret = pycv_morphology.gray_ero_or_dil(0, image, strel, offset, mask, output, border_val)
    return output if ret is None else ret


def gray_dilation(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
        border_val: int = 0
) -> np.ndarray:
    ret = pycv_morphology.gray_ero_or_dil(1, image, strel, offset, mask, output, border_val)
    return output if ret is None else ret


def gray_opening(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
        border_val: int = 0
) -> np.ndarray:
    ero = pycv_morphology.gray_ero_or_dil(0, image, strel, offset, mask, None, border_val)
    ret = pycv_morphology.gray_ero_or_dil(1, ero, strel, offset, mask, output, border_val)
    return output if ret is None else ret


def gray_closing(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
        border_val: int = 0
) -> np.ndarray:
    dil = pycv_morphology.gray_ero_or_dil(1, image, strel, offset, mask, None, border_val)
    ret = pycv_morphology.gray_ero_or_dil(0, dil, strel, offset, mask, output, border_val)
    return output if ret is None else ret


########################################################################################################################

def black_top(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
        border_val: int = 0
) -> np.ndarray:
    cl = gray_closing(image, strel, offset=offset, mask=mask, output=output, border_val=border_val)
    cl -= image
    return cl


def white_top(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None,
        border_val: int = 0
) -> np.ndarray:
    op = gray_opening(image, strel, offset=offset, mask=mask, output=output, border_val=border_val)
    op = image - op
    return op


########################################################################################################################

def area_open(
        image: np.ndarray,
        threshold: int = 32,
        connectivity: int = 1,
) -> np.ndarray:
    return pycv_morphology.area_open_close('open', image, threshold=threshold, connectivity=connectivity)


def area_close(
        image: np.ndarray,
        threshold: int = 32,
        connectivity: int = 1,
) -> np.ndarray:
    return pycv_morphology.area_open_close('close', image, threshold=threshold, connectivity=connectivity)

########################################################################################################################
