import numpy as np
from pycv._lib.filters_support.morphology import c_gray_ero_or_dil

__all__ = [
    'gray_erosion',
    'gray_dilation',
    'gray_opening',
    'gray_closing',
    'black_top',
    'white_top',
    'PUBLIC'
]

PUBLIC = [
    'gray_erosion',
    'gray_dilation',
    'gray_opening',
    'gray_closing',
    'black_top',
    'white_top',
]


########################################################################################################################

def gray_erosion(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        iterations: int = 1,  # TODO: iterations
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    ret = c_gray_ero_or_dil(0, image, strel, offset, iterations, mask, output)
    return output if ret is None else ret


def gray_dilation(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        iterations: int = 1,  # TODO: iterations
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    ret = c_gray_ero_or_dil(1, image, strel, offset, iterations, mask, output)
    return output if ret is None else ret


def gray_opening(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    ero = c_gray_ero_or_dil(0, image, strel, offset, iterations, mask, None)
    ret = c_gray_ero_or_dil(1, ero, strel, offset, iterations, mask, output)
    return output if ret is None else ret


def gray_closing(
        image: np.ndarray,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        mask: np.ndarray | None = None,
        output: np.ndarray | None = None
) -> np.ndarray:
    dil = c_gray_ero_or_dil(1, image, strel, offset, iterations, mask, None)
    ret = c_gray_ero_or_dil(0, dil, strel, offset, iterations, mask, output)
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
