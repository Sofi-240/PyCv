import numpy as np
import typing
from .._lib.array_api.dtypes import get_dtype_limits
from .._lib.filters_support._windows import square, circle, cross, diamond
from .._lib._members_struct import Members

__all__ = [
    'Shapes',
    'mark_points'
]


########################################################################################################################

class Shapes(Members):
    SQUARE = 1
    CROSS = 2
    DIAMOND = 3
    CIRCLE = 4


def _get_shape(shape: Shapes | str | int, min_axis: int) -> np.ndarray:
    if shape == Shapes.SQUARE:
        return square(min(min_axis * 2 + 1, 7))
    elif shape == Shapes.CROSS:
        return cross((min(min_axis * 2 + 1, 7), ) * 2)
    elif shape == Shapes.DIAMOND:
        return diamond(min((min_axis * 2 + 1) // 2, 3))
    elif shape == Shapes.CIRCLE:
        return circle(min((min_axis * 2 + 1) // 2, 3))
    else:
        raise TypeError('shape is not member of Shapes')


########################################################################################################################


def mark_points(
        image: np.ndarray,
        points: typing.Iterable,
        shape: Shapes | str | int = Shapes.CIRCLE,
        color: tuple | None = None
) -> np.ndarray:
    output = np.asarray(image)
    if output.ndim not in (2, 3):
        raise ValueError('image need to be 2D or 3D array')

    if not isinstance(points, np.ndarray):
        points = np.array(points, np.int64)
    else:
        points = points.astype(np.int64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('invalid points shape expected to be (n points, 2)')

    if output.ndim == 2:
        output = np.stack([output] * 3, axis=-1)

    if output.dtype == bool:
        output = output.astype(np.uint8) * 255
        c = 255
    elif output.dtype.kind == 'f':
        c = max(np.max(output), 1)
    else:
        c = get_dtype_limits(output.dtype)[1]

    if color is None:
        color = (c, 0, 0)
    elif len(color) != 3:
        raise ValueError('color need to be in RGB format')

    color = np.array(color, dtype=output.dtype)

    for point in points:
        ii, jj = point
        if ii < 0 or ii >= output.shape[0] or jj < 0 or jj >= output.shape[1]:
            raise ValueError(f'point: {point} is out of range for image with shape of {output.shape}')
        sh = _get_shape(shape, min(ii, jj))
        r = sh.shape[0] // 2
        tmp = output[ii - r:ii + r + 1, jj - r:jj + r + 1, :]

        tmp[..., :][sh] = color

    return output

########################################################################################################################
