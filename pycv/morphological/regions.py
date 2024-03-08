import numpy as np
from pycv._lib._src_py import pycv_morphology, pycv_convexhull
from pycv._lib.array_api.regulator import np_compliance

__all__ = [
    'region_fill',
    'im_label',
    'convex_hull',
    'convex_image',
    'find_object'
]


########################################################################################################################

def region_fill(
        image: np.ndarray,
        seed_point: tuple,
        strel: np.ndarray | None = None,
        offset: tuple | None = None,
        output: np.ndarray | None = None,
        inplace: bool = False,
        value_tol: int | float = 0,
        fill_value: int | float | None = None
) -> np.ndarray:
    """
    Fill a region of interest in the input image starting from a specified seed point.

    This function fills a connected region in the input image starting from the seed point. It supports both binary and grayscale images. For binary images, the region is filled with True values. For grayscale images, the region is filled with the specified fill value or the value at the seed point.

    Parameters:
        image (numpy.ndarray): Input image to be filled.
        seed_point (tuple): Seed point (coordinates) from which the filling starts. It should be a tuple of coordinates, with one value for each dimension of the image.
        strel (numpy.ndarray or None, optional): Structuring element used for the region filling. If None, a structuring element with a square shape (3x3) will be used. Defaults to None.
        offset (tuple or None, optional): Offset for the structuring element. Defaults to None.
        output (numpy.ndarray or None, optional): Output array to store the result of the operation. If None, a new array will be created. Defaults to None.
        inplace (bool, optional): If True, the operation will be performed in place on the input image. Defaults to False.
        value_tol (int or float, optional): Tolerance value for considering nearby pixels during region filling. Defaults to 0.
        fill_value (int, float, or None, optional): Value used to fill the region in grayscale images. If None, the value at the seed point will be used. Defaults to None.

    Returns:
        numpy.ndarray: Output image after applying the region filling operation.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f'Image need to be type of numpy.ndarray')

    if len(seed_point) != image.ndim:
        raise ValueError('Number of dimensions in seed_point and img do not match')

    if not all(0 <= sp < s for sp, s in zip(seed_point, image.shape)):
        raise ValueError('Seed point is out of range')

    if image.dtype == bool:
        return pycv_morphology.binary_region_fill(image, seed_point, strel, offset, output, inplace)

    seed_value = image[seed_point]

    inputs = np.where((image >= seed_value - value_tol) & (image <= seed_value + value_tol), False, True)

    pycv_morphology.binary_region_fill(inputs, seed_point, strel, offset, None, True)

    if inplace:
        output = image
    elif not output:
        output = np.zeros_like(image)

    output[inputs] = fill_value if fill_value is not None else seed_value

    return output


########################################################################################################################

def im_label(
        image: np.ndarray,
        connectivity: int = 1,
        rng_mapping_method: str = 'sqr',
        mod_value: int = 16
) -> tuple[int, np.ndarray]:
    """
    Label connected components in the input binary image.

    This function assigns a unique label to each connected component in the input binary image. It supports both 4-connectivity and 8-connectivity.

    Parameters:
        image (numpy.ndarray): Input binary image to be labeled.
        connectivity (int, optional): Connectivity of the connected components. It can be 1 for 4-connectivity or 2 for 8-connectivity. Defaults to 1.
        rng_mapping_method (str, optional): Method used for generating label values. It can be 'sqr' for square root or 'log' for logarithmic. Defaults to 'sqr'.
        mod_value (int, optional): Modulation value used in label generation. Defaults to 16.

    Returns:
        tuple[int, numpy.ndarray]: A tuple containing the number of connected components and the labeled image where each pixel is assigned a unique label.

    Raises:
        ValueError: If an unsupported connectivity or rng_mapping_method is provided.
    """
    return pycv_morphology.labeling(image, connectivity, rng_mapping_method, mod_value)


########################################################################################################################


def convex_hull(
        image: np.ndarray,
        mask: np.ndarray | None = None,
        objects: bool = False,
        labels: np.ndarray | None = None,
        convex_img: bool = True
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Calculate the convex hull of objects in the input binary image.

    This function computes the convex hull of objects in the input binary image. It can operate on the entire image or individual objects if 'objects' is set to True.

    Parameters:
        image (numpy.ndarray): Input binary image.
        mask (numpy.ndarray or None, optional): Mask to limit the operation to a specific area. If None, the entire image will be processed. Defaults to None.
        objects (bool, optional): If True, the function operates on individual objects. Defaults to False.
        labels (numpy.ndarray or None, optional): Label image specifying objects if 'objects' is set to True. Defaults to None.
        convex_img (bool, optional): If True, returns the convex hull image. If False, returns the vertices of the convex hull. Defaults to True.

    Returns:
        numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]: If 'convex_img' is True, returns the convex hull image. If False, returns a tuple containing the vertices of the convex hull and the corresponding labels.

    Raises:
        TypeError: If the labels image is of type float.
        ValueError: If the mask shape does not match the labels shape.

    Notes:
        - If 'objects' is True, 'labels' must be provided.
    """
    if objects:
        if labels is None:
            _, labels = im_label(image)
        if labels.dtype.kind == 'f':
            raise TypeError('labels image cannot be type of float')
        uni = np.unique(labels[labels != 0])
        if mask is None:
            mask = np.ones_like(labels, bool)
        elif mask.shape != labels.shape:
            raise ValueError('mask shape need to be same as labels shape')
        inputs = np.stack([np.asarray((labels == u) & mask, dtype=np.uint8) for u in uni], axis=0)
        mask = None
    else:
        inputs = image

    return pycv_convexhull.convex_hull_2d(inputs, mask, convex_image=convex_img)


def convex_image(
        convex_hull_points: np.ndarray,
        output_shape: tuple | None = None,
        axis: tuple | None = None
) -> np.ndarray:
    """
    Generate a binary image from the given convex hull points.

    This function creates a binary image from the specified convex hull points. It can be used to visualize the convex hull.

    Parameters:
        convex_hull_points (numpy.ndarray): Points defining the convex hull.
        output_shape (tuple or None, optional): Shape of the output image. If None, the shape is determined automatically. Defaults to None.
        axis (tuple or None, optional): Axis along which the points represent coordinates. If None, all axes are considered. Defaults to None.

    Returns:
        numpy.ndarray: Binary image representing the convex hull.

    Notes:
        - The 'convex_hull_points' should be a 2D array where each row contains the coordinates of a point.
        - If 'output_shape' is None, it is determined based on the maximum coordinates of the points.
        - 'axis' specifies which axes of the input points correspond to spatial dimensions.
    """
    return pycv_convexhull.convex_hull_2d_image(convex_hull_points, output_shape, axis)


########################################################################################################################


def find_object(
        labels: np.ndarray,
        mask: np.ndarray | None = None,
        as_slice: bool = False,
) -> list[tuple]:
    """
    Find the bounding boxes or slices of connected components in a labeled image.

    This function identifies connected components in the labeled image and returns their bounding boxes or slices.

    Parameters:
        labels (numpy.ndarray): Labeled image where connected components are identified.
        mask (numpy.ndarray or None, optional): Mask to apply on the labeled image before finding objects.
            If provided, only the labeled regions within the mask are considered. Defaults to None.
        as_slice (bool, optional): If True, returns bounding boxes as slices.
            If False, returns bounding boxes as pairs of top-left and bottom-right coordinates. Defaults to False.

    Returns:
        list[tuple]: List of bounding boxes or slices representing connected components.

    Notes:
        - The 'labels' array should be an integer-labeled image where connected components are marked with unique integers.
        - If 'mask' is provided, only the labeled regions within the mask are considered.
        - If 'as_slice' is True, the function returns bounding boxes as slices, otherwise as pairs of coordinates.
    """
    labels = np_compliance(labels, 'labels')
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('labels dtype need to be integer')
    ll_im = labels.copy()
    if mask is not None:
        if mask.shape != labels.shape:
            raise ValueError('mask shape need to be same as labels shape')
        ll_im[~mask] = 0

    out = []

    for lbl in np.unique(ll_im[ll_im > 0]):
        if as_slice:
            out.append(tuple(slice(np.amin(cc), np.amax(cc) + 1) for cc in np.where(ll_im == lbl)))
        else:
            cc = np.stack(np.where(ll_im == lbl), axis=1)
            out.append((np.amin(cc, axis=0), np.amax(cc, axis=0)))

    return out

########################################################################################################################
