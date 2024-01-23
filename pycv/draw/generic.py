import numpy as np
from pycv._lib.array_api.dtypes import get_dtype_limits
from pycv._lib.core_support import image_support_py

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
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    if output is None:
        return image_support_py.draw('line', point1=point1, point2=point2)

    if not isinstance(output, np.ndarray):
        raise TypeError('output need to be type of numpy.ndarray')

    shape = output.shape
    if output.ndim < 2:
        raise ValueError('expected output rank to be atleast 2')

    if any(p < 0 or p >= shape[-2] for p in (point1[0], point2[0])) or any(
            p < 0 or p >= shape[-1] for p in (point1[1], point2[1])):
        raise ValueError("points is out of range for output shape")

    coord = image_support_py.draw('line', point1=point1, point2=point2)

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
    if output is None:
        return image_support_py.draw('circle', center_point=center_point, radius=radius)

    if not isinstance(output, np.ndarray):
        raise TypeError('output need to be type of numpy.ndarray')

    shape = output.shape
    if output.ndim < 2:
        raise ValueError('expected output rank to be atleast 2')

    if any(p < 0 or p >= s or p - radius <= 0 or p + radius >= s for p, s in zip(center_point, shape)):
        raise ValueError('point or radius shift is out of range for output shape')

    coord = image_support_py.draw('circle', center_point=center_point, radius=radius)

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
    if output is None:
        return image_support_py.draw('ellipse', center_point=center_point, a=a, b=b)

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

    coord = image_support_py.draw('ellipse', center_point=center_point, a=a, b=b)

    if fill_value is None:
        fill_value = get_dtype_limits(output.dtype)[1]

    if output.ndim == 2:
        output[coord] = fill_value
    else:
        output[(Ellipsis,) + coord] = fill_value

    return output
