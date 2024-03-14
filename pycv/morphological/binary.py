import numpy as np
from .._lib._src_py import pycv_morphology

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
    """
    Apply binary erosion operation to the input binary image.

    Binary erosion is a morphological operation that shrinks the boundaries of regions of foreground pixels (i.e., pixels with value 1) in a binary image.

    Parameters:
        image (numpy.ndarray): Input binary image to which the binary erosion operation will be applied.
        strel (numpy.ndarray or None, optional): Structuring element used for erosion. If None, a structuring element with a square shape (3x3) will be used. Defaults to None.
        offset (tuple or None, optional): Offset for the structuring element. Defaults to None.
        iterations (int, optional): Number of iterations for the erosion operation. Defaults to 1.
        mask (numpy.ndarray or None, optional): Mask array of the same shape as the input image. If provided, only the pixels where the mask value is non-zero will be considered for erosion. Defaults to None.
        output (numpy.ndarray or None, optional): Output array to store the result of the erosion operation. If None, a new array will be created. Defaults to None.
        border_val (int, optional): Value used for padding at the image border if extra_memory is True. Defaults to 0.
        extra_memory (bool, optional): If True, extra memory will be used for computation. Defaults to True.

    Returns:
        numpy.ndarray: Output binary image after applying the binary erosion operation.
    """
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
    """
    Apply binary dilation operation to the input binary image.

    Binary dilation is a morphological operation that expands the boundaries of regions of foreground pixels (i.e., pixels with value 1) in a binary image.

    Parameters:
        image (numpy.ndarray): Input binary image to which the binary dilation operation will be applied.
        strel (numpy.ndarray or None, optional): Structuring element used for dilation. If None, a structuring element with a square shape (3x3) will be used. Defaults to None.
        offset (tuple or None, optional): Offset for the structuring element. Defaults to None.
        iterations (int, optional): Number of iterations for the dilation operation. Defaults to 1.
        mask (numpy.ndarray or None, optional): Mask array of the same shape as the input image. If provided, only the pixels where the mask value is non-zero will be considered for dilation. Defaults to None.
        output (numpy.ndarray or None, optional): Output array to store the result of the dilation operation. If None, a new array will be created. Defaults to None.
        border_val (int, optional): Value used for padding at the image border if extra_memory is True. Defaults to 0.
        extra_memory (bool, optional): If True, extra memory will be used for computation. Defaults to True.

    Returns:
        numpy.ndarray: Output binary image after applying the binary dilation operation.
    """
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
    """
    Apply binary opening operation to the input binary image.

    Binary opening is a morphological operation that performs binary erosion followed by binary dilation on a binary image.

    Parameters:
        image (numpy.ndarray): Input binary image to which the binary opening operation will be applied.
        strel (numpy.ndarray or None, optional): Structuring element used for binary opening. If None, a structuring element with a square shape (3x3) will be used. Defaults to None.
        offset (tuple or None, optional): Offset for the structuring element. Defaults to None.
        mask (numpy.ndarray or None, optional): Mask array of the same shape as the input image. If provided, only the pixels where the mask value is non-zero will be considered for binary opening. Defaults to None.
        output (numpy.ndarray or None, optional): Output array to store the result of the binary opening operation. If None, a new array will be created. Defaults to None.
        border_val (int, optional): Value used for padding at the image border. Defaults to 0.

    Returns:
        numpy.ndarray: Output binary image after applying the binary opening operation.
    """
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
    """
    Apply binary closing operation to the input binary image.

    Binary closing is a morphological operation that first applies binary dilation and then binary erosion to a binary image, resulting in smoothing or filling small holes in the foreground regions.

    Parameters:
        image (numpy.ndarray): Input binary image to which the binary closing operation will be applied.
        strel (numpy.ndarray or None, optional): Structuring element used for the closing operation. If None, a structuring element with a square shape (3x3) will be used. Defaults to None.
        offset (tuple or None, optional): Offset for the structuring element. Defaults to None.
        mask (numpy.ndarray or None, optional): Mask array of the same shape as the input image. If provided, only the pixels where the mask value is non-zero will be considered for the closing operation. Defaults to None.
        output (numpy.ndarray or None, optional): Output array to store the result of the closing operation. If None, a new array will be created. Defaults to None.
        border_val (int, optional): Value used for padding at the image border. Defaults to 0.

    Returns:
        numpy.ndarray: Output binary image after applying the binary closing operation.
    """
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
    """
    Detect edges in the input binary image using morphological operations.

    Binary edge detection identifies the boundaries between regions of foreground pixels (value 1) and background pixels (value 0) in a binary image.

    Parameters:
        image (numpy.ndarray): Input binary image in which edges will be detected.
        edge_mode (str, optional): Mode of edge detection. Possible values are 'inner', 'outer', or 'double'. Defaults to 'inner'.
        strel (numpy.ndarray or None, optional): Structuring element used for edge detection. If None, a structuring element with a square shape (3x3) will be used. Defaults to None.
        offset (tuple or None, optional): Offset for the structuring element. Defaults to None.
        mask (numpy.ndarray or None, optional): Mask array of the same shape as the input image. If provided, only the pixels where the mask value is non-zero will be considered for edge detection. Defaults to None.
        output (numpy.ndarray or None, optional): Output array to store the result of edge detection. If None, a new array will be created. Defaults to None.
        border_val (int, optional): Value used for padding at the image border. Defaults to 0.

    Returns:
        numpy.ndarray: Output binary image containing the detected edges.
        """
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
    """
    Compute the skeleton of a binary image.

    Skeletonization is a morphological operation that reduces binary structures into their skeletal representation, which typically consists of center lines or curves.

    Parameters:
        image (numpy.ndarray): Input binary image to compute the skeleton from.

    Returns:
        numpy.ndarray: Skeletonized binary image.
    """
    return pycv_morphology.skeletonize(image)


########################################################################################################################

def remove_small_objects(
        image: np.ndarray,
        threshold: int = 32,
        connectivity: int = 1
) -> np.ndarray:
    """
    Remove small connected components from a binary image.

    This function removes connected components (objects) from a binary image that have fewer than a specified number of pixels.

    Parameters:
        image (numpy.ndarray): Input binary image from which small objects will be removed.
        threshold (int, optional): Minimum number of pixels required for an object to be retained. Defaults to 32.
        connectivity (int, optional): Connectivity of the objects. Defaults to 1.

    Returns:
        numpy.ndarray: Binary image with small objects removed.
    """
    return pycv_morphology.remove_small_objects(image, threshold, connectivity)


def remove_small_holes(
        image: np.ndarray,
        threshold: int = 32,
        connectivity: int = 1
) -> np.ndarray:
    """
    Remove small holes (foreground regions) from a binary image.

    This function removes small holes (foreground regions) from a binary image that have fewer than a specified number of pixels.

    Parameters:
        image (numpy.ndarray): Input binary image from which small holes will be removed.
        threshold (int, optional): Minimum number of pixels required for a hole to be retained. Defaults to 32.
        connectivity (int, optional): Connectivity of the holes. Defaults to 1.

    Returns:
        numpy.ndarray: Binary image with small holes removed.
    """
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
    """
    Apply binary hit-or-miss transform to the input binary image.

    The hit-or-miss transform is a morphological operation used to detect simple shapes (patterns) in binary images.

    Parameters:
        image (numpy.ndarray): Input binary image to which the hit-or-miss transform will be applied.
        strel1 (numpy.ndarray or None, optional): Structuring element for the "hit" part of the hit-or-miss transform. If None, a structuring element with a square shape (3x3) will be used. Defaults to None.
        strel2 (numpy.ndarray or None, optional): Structuring element for the "miss" part of the hit-or-miss transform. If None, a structuring element with a square shape (3x3) will be used. Defaults to None.
        offset1 (tuple or None, optional): Offset for the "hit" structuring element. Defaults to None.
        offset2 (tuple or None, optional): Offset for the "miss" structuring element. Defaults to None.
        mask (numpy.ndarray or None, optional): Mask array of the same shape as the input image. If provided, only the pixels where the mask value is non-zero will be considered for the hit-or-miss transform. Defaults to None.
        output (numpy.ndarray or None, optional): Output array to store the result of the hit-or-miss transform. If None, a new array will be created. Defaults to None.
        border_val (int, optional): Value used for padding at the image border. Defaults to 0.

    Returns:
        numpy.ndarray: Output binary image after applying the hit-or-miss transform.
    """
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
    """
    Fill holes in the foreground regions of a binary image.

    This function fills holes (connected components with value 0 surrounded by foreground pixels with value 1) in the foreground regions of a binary image.

    Parameters:
        image (numpy.ndarray): Input binary image in which holes will be filled.
        strel (numpy.ndarray or None, optional): Structuring element used for the filling operation. If None, a structuring element with a square shape (3x3) will be used. Defaults to None.
        offset (tuple or None, optional): Offset for the structuring element. Defaults to None.
        output (numpy.ndarray or None, optional): Output array to store the result of the filling operation. If None, a new array will be created. Defaults to None.
        extra_memory (bool, optional): If True, extra memory will be used for computation. Defaults to True.

    Returns:
        numpy.ndarray: Binary image with holes filled.
    """
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
