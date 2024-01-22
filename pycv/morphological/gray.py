import numpy as np
from pycv._lib.core_support import morphology_py

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
        output: np.ndarray | None = None
) -> np.ndarray:
    ret = morphology_py.gray_ero_or_dil(0, image, strel, offset, mask, output)
    return output if ret is None else ret


def gray_dilation(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    ret = morphology_py.gray_ero_or_dil(1, image, strel, offset, mask, output)
    return output if ret is None else ret


def gray_opening(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    ero = morphology_py.gray_ero_or_dil(0, image, strel, offset, mask, None)
    ret = morphology_py.gray_ero_or_dil(1, ero, strel, offset, mask, output)
    return output if ret is None else ret


def gray_closing(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    dil = morphology_py.gray_ero_or_dil(1, image, strel, offset, mask, None)
    ret = morphology_py.gray_ero_or_dil(0, dil, strel, offset, mask, output)
    return output if ret is None else ret


########################################################################################################################

def black_top(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    cl = gray_closing(image, strel, offset=offset, mask=mask, output=output)
    cl -= image
    return cl


def white_top(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    op = gray_opening(image, strel, offset=offset, mask=mask, output=output)
    op = image - op
    return op


########################################################################################################################

def area_open(
        image: np.ndarray,
        threshold: int = 32,
        connectivity: int = 1,
) -> np.ndarray:
    return morphology_py.area_open_close('open', image, threshold=threshold, connectivity=connectivity)


def area_close(
        image: np.ndarray,
        threshold: int = 32,
        connectivity: int = 1,
) -> np.ndarray:
    return morphology_py.area_open_close('close', image, threshold=threshold, connectivity=connectivity)
