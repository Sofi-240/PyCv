import numpy as np
from .._lib.array_api.dtypes import get_dtype_limits
from .._lib._src_py import pycv_minsc

__all__ = [
    'draw_line',
    'draw_circle',
    'draw_ellipse',
]


########################################################################################################################

def draw_line(
        point1: tuple,
        point2: tuple,
        output: np.ndarray | None = None,
        fill_value: int | float | None = None
) -> np.ndarray:
    """
    Draw line image. if output is None then return the points of the line else draw the line on the output image.

    Parameters
    ----------
    point1 : tuple
        coordinates of the first point.
    point2 : tuple
        coordinates of the second point.
    output : np.ndarray | None
        default is None. if the output is given the line will be drawn on the output
    fill_value : int | float | None
        The value of the line in the output image.
        if the output image is None ignor this parameter.
        default the maximum value of the output dtype.

    Returns
    -------
    points coordinates or the output image: numpy.ndarray


    Raises
    ------
    TypeError
        If the output is not type of numpy.ndarray.
    ValueError
        If the output ndim is smaller than 2.
    ValueError
        If the points is out of range for the output shape
    """
    if output is None:
        return pycv_minsc.draw('line', point1=point1, point2=point2)

    if not isinstance(output, np.ndarray):
        raise TypeError('output need to be type of numpy.ndarray')

    shape = output.shape
    if output.ndim < 2:
        raise ValueError('expected output rank to be atleast 2')

    if any(p < 0 or p >= shape[-2] for p in (point1[0], point2[0])) or any(
            p < 0 or p >= shape[-1] for p in (point1[1], point2[1])):
        raise ValueError("points is out of range for output shape")

    coord = pycv_minsc.draw('line', point1=point1, point2=point2)

    if fill_value is None:
        fill_value = get_dtype_limits(output.dtype)[1]

    if output.ndim == 2:
        output[coord] = fill_value
    else:
        output[(Ellipsis,) + coord] = fill_value

    return output


def draw_circle(
        center_point: tuple,
        radius: int,
        output: np.ndarray | None = None,
        fill_value: int | float | None = None
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Draw circle image. if output is None then return the points of the circle else draw the circle on the output image.

    Parameters
    ----------
    center_point : tuple
        coordinates of the circle center.
    radius : int
        circle radius.
    output : np.ndarray | None
        default is None. if the output is given the circle will be drawn on the output
    fill_value : int | float | None
        The value of the circle in the output image.
        if the output image is None ignor this parameter.
        default the maximum value of the output dtype.

    Returns
    -------
    points coordinates or the output image: numpy.ndarray


    Raises
    ------
    TypeError
        If the output is not type of numpy.ndarray.
    ValueError
        If the output ndim is smaller than 2.
    ValueError
        If the points or the points with the radius is out of range for the output shape
    """
    if output is None:
        return pycv_minsc.draw('circle', center_point=center_point, radius=radius)

    if not isinstance(output, np.ndarray):
        raise TypeError('output need to be type of numpy.ndarray')

    shape = output.shape
    if output.ndim < 2:
        raise ValueError('expected output rank to be atleast 2')

    if any(p < 0 or p >= s or p - radius <= 0 or p + radius >= s for p, s in zip(center_point, shape)):
        raise ValueError('point or radius shift is out of range for output shape')

    coord = pycv_minsc.draw('circle', center_point=center_point, radius=radius)

    if fill_value is None:
        fill_value = get_dtype_limits(output.dtype)[1]

    if output.ndim == 2:
        output[coord] = fill_value
    else:
        output[(Ellipsis,) + coord] = fill_value

    return output


def draw_ellipse(
        center_point: tuple,
        a: int,
        b: int,
        output: np.ndarray | None = None,
        fill_value: int | float | None = None
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Draw ellipse image. if output is None then return the points of the ellipse else draw the ellipse on the output image.

    Parameters
    ----------
    center_point : tuple
        coordinates of the circle center.
    a : int
        major axis radius.
    b : int
        minor axis radius.
    output : np.ndarray | None
        default is None. if the output is given the ellipse will be drawn on the output
    fill_value : int | float | None
        The value of the ellipse in the output image.
        if the output image is None ignor this parameter.
        default the maximum value of the output dtype.

    Returns
    -------
    points coordinates or the output image: numpy.ndarray


    Raises
    ------
    TypeError
        If the output is not type of numpy.ndarray.
    ValueError
        If the output ndim is smaller than 2.
    ValueError
        If the points or the points with the radius is out of range for the output shape
    """
    if output is None:
        return pycv_minsc.draw('ellipse', center_point=center_point, a=a, b=b)

    if not isinstance(output, np.ndarray):
        raise TypeError('output need to be type of numpy.ndarray')

    shape = output.shape
    if output.ndim < 2:
        raise ValueError('expected output rank to be atleast 2')

    if any(p < 0 or p >= s for p, s in zip(center_point, shape)):
        raise ValueError('point is out of range for output shape')

    if center_point[0] - a < 0 or center_point[0] + a >= shape[0]:
        raise ValueError('a shift is out of range for output shape')

    if center_point[1] - b < 0 or center_point[1] + b >= shape[1]:
        raise ValueError('b shift is out of range for output shape')

    coord = pycv_minsc.draw('ellipse', center_point=center_point, a=a, b=b)

    if fill_value is None:
        fill_value = get_dtype_limits(output.dtype)[1]

    if output.ndim == 2:
        output[coord] = fill_value
    else:
        output[(Ellipsis,) + coord] = fill_value

    return output


########################################################################################################################