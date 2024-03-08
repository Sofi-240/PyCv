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
    """
    Apply grayscale erosion operation to the input grayscale image.

    Grayscale erosion is a morphological operation that computes the local minimum over the neighborhood defined by a structuring element.

    Parameters:
        image (numpy.ndarray): Input grayscale image to which the grayscale erosion operation will be applied.
        strel (numpy.ndarray or None, optional): Structuring element used for erosion. If None, a structuring element with a square shape (3x3) will be used. Defaults to None.
        offset (tuple or None, optional): Offset for the structuring element. Defaults to None.
        mask (numpy.ndarray or None, optional): Mask array of the same shape as the input image. If provided, only the pixels where the mask value is non-zero will be considered for erosion. Defaults to None.
        output (numpy.ndarray or None, optional): Output array to store the result of the erosion operation. If None, a new array will be created. Defaults to None.
        border_val (int, optional): Value used for padding at the image border. Defaults to 0.

    Returns:
        numpy.ndarray: Output grayscale image after applying the erosion operation.
    """
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
    """
    Apply grayscale dilation operation to the input grayscale image.

    Grayscale dilation is a morphological operation that computes the local maximum over the neighborhood defined by a structuring element.

    Parameters:
        image (numpy.ndarray): Input grayscale image to which the grayscale dilation operation will be applied.
        strel (numpy.ndarray or None, optional): Structuring element used for dilation. If None, a structuring element with a square shape (3x3) will be used. Defaults to None.
        offset (tuple or None, optional): Offset for the structuring element. Defaults to None.
        mask (numpy.ndarray or None, optional): Mask array of the same shape as the input image. If provided, only the pixels where the mask value is non-zero will be considered for dilation. Defaults to None.
        output (numpy.ndarray or None, optional): Output array to store the result of the dilation operation. If None, a new array will be created. Defaults to None.
        border_val (int, optional): Value used for padding at the image border. Defaults to 0.

    Returns:
        numpy.ndarray: Output grayscale image after applying the dilation operation.
    """
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
    """
    Apply grayscale opening operation to the input grayscale image.

    Grayscale opening is a morphological operation that first applies grayscale erosion followed by grayscale dilation.

    Parameters:
        image (numpy.ndarray): Input grayscale image to which the grayscale opening operation will be applied.
        strel (numpy.ndarray or None, optional): Structuring element used for opening. If None, a structuring element with a square shape (3x3) will be used. Defaults to None.
        offset (tuple or None, optional): Offset for the structuring element. Defaults to None.
        mask (numpy.ndarray or None, optional): Mask array of the same shape as the input image. If provided, only the pixels where the mask value is non-zero will be considered for opening. Defaults to None.
        output (numpy.ndarray or None, optional): Output array to store the result of the opening operation. If None, a new array will be created. Defaults to None.
        border_val (int, optional): Value used for padding at the image border. Defaults to 0.

    Returns:
        numpy.ndarray: Output grayscale image after applying the opening operation.
    """
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
    """
    Apply grayscale closing operation to the input grayscale image.

    Grayscale closing is a morphological operation that first applies grayscale dilation followed by grayscale erosion.

    Parameters:
        image (numpy.ndarray): Input grayscale image to which the grayscale closing operation will be applied.
        strel (numpy.ndarray or None, optional): Structuring element used for closing. If None, a structuring element with a square shape (3x3) will be used. Defaults to None.
        offset (tuple or None, optional): Offset for the structuring element. Defaults to None.
        mask (numpy.ndarray or None, optional): Mask array of the same shape as the input image. If provided, only the pixels where the mask value is non-zero will be considered for closing. Defaults to None.
        output (numpy.ndarray or None, optional): Output array to store the result of the closing operation. If None, a new array will be created. Defaults to None.
        border_val (int, optional): Value used for padding at the image border. Defaults to 0.

    Returns:
        numpy.ndarray: Output grayscale image after applying the closing operation.
    """
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
    """
    Apply black top hat operation to the input grayscale image.

    Black top hat is a morphological operation that computes the difference between the input image and the closing of the input image.

    Parameters:
        image (numpy.ndarray): Input grayscale image to which the black top hat operation will be applied.
        strel (numpy.ndarray or None, optional): Structuring element used for the closing operation. If None, a structuring element with a square shape (3x3) will be used. Defaults to None.
        offset (tuple or None, optional): Offset for the structuring element. Defaults to None.
        mask (numpy.ndarray or None, optional): Mask array of the same shape as the input image. If provided, only the pixels where the mask value is non-zero will be considered for the operation. Defaults to None.
        output (numpy.ndarray or None, optional): Output array to store the result of the operation. If None, a new array will be created. Defaults to None.
        border_val (int, optional): Value used for padding at the image border. Defaults to 0.

    Returns:
        numpy.ndarray: Output grayscale image after applying the black top hat operation.
    """
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
    """
    Apply white top hat operation to the input grayscale image.

    White top hat is a morphological operation that computes the difference between the opening of the input image and the input image.

    Parameters:
        image (numpy.ndarray): Input grayscale image to which the white top hat operation will be applied.
        strel (numpy.ndarray or None, optional): Structuring element used for the opening operation. If None, a structuring element with a square shape (3x3) will be used. Defaults to None.
        offset (tuple or None, optional): Offset for the structuring element. Defaults to None.
        mask (numpy.ndarray or None, optional): Mask array of the same shape as the input image. If provided, only the pixels where the mask value is non-zero will be considered for the operation. Defaults to None.
        output (numpy.ndarray or None, optional): Output array to store the result of the operation. If None, a new array will be created. Defaults to None.
        border_val (int, optional): Value used for padding at the image border. Defaults to 0.

    Returns:
        numpy.ndarray: Output grayscale image after applying the white top hat operation.
    """
    op = gray_opening(image, strel, offset=offset, mask=mask, output=output, border_val=border_val)
    op = image - op
    return op


########################################################################################################################

def area_open(
        image: np.ndarray,
        threshold: int = 32,
        connectivity: int = 1,
) -> np.ndarray:
    """
    Apply area opening operation to the input binary image.

    Area opening is a morphological operation that removes connected components (objects) from the binary image that have an area smaller than the specified threshold.

    Parameters:
        image (numpy.ndarray): Input binary image to which the area opening operation will be applied.
        threshold (int, optional): Minimum area (number of pixels) of connected components to be retained. Objects with smaller areas will be removed. Defaults to 32.
        connectivity (int, optional): Connectivity of the objects. It can be 1 for 4-connectivity or 2 for 8-connectivity. Defaults to 1.

    Returns:
        numpy.ndarray: Output binary image after applying the area opening operation.
    """
    return pycv_morphology.area_open_close('open', image, threshold=threshold, connectivity=connectivity)


def area_close(
        image: np.ndarray,
        threshold: int = 32,
        connectivity: int = 1,
) -> np.ndarray:
    """
    Apply area closing operation to the input binary image.

    Area closing is a morphological operation that fills holes and connects gaps in connected components (objects) of the binary image that have an area smaller than the specified threshold.

    Parameters:
        image (numpy.ndarray): Input binary image to which the area closing operation will be applied.
        threshold (int, optional): Minimum area (number of pixels) of connected components to be filled or connected. Objects with smaller areas will be affected. Defaults to 32.
        connectivity (int, optional): Connectivity of the objects. It can be 1 for 4-connectivity or 2 for 8-connectivity. Defaults to 1.

    Returns:
        numpy.ndarray: Output binary image after applying the area closing operation.
    """
    return pycv_morphology.area_open_close('close', image, threshold=threshold, connectivity=connectivity)

########################################################################################################################
