import numpy as np
from .utils import get_output, valid_same_shape
from pycv._lib._src import c_pycv

__all__ = [
    'canny_nonmaximum_suppression',
    'draw',
]


########################################################################################################################

def canny_nonmaximum_suppression(
        magnitude: np.ndarray,
        grad_y: np.ndarray,
        grad_x: np.ndarray,
        low_threshold: float,
        mask: np.ndarray | None
) -> np.ndarray:
    valid_shape_tuple = (magnitude, grad_y, grad_x)
    valid_shape_tuple += (mask,) if mask is not None else tuple()
    if not valid_same_shape(*valid_shape_tuple):
        raise RuntimeError('all the ndarray inputs need to have the same shape')
    output, _ = get_output(None, magnitude)
    c_pycv.canny_nonmaximum_suppression(magnitude, grad_y, grad_x, low_threshold, mask, output)
    return output


########################################################################################################################

def draw(
        mode: str,
        point1: tuple | None = None,
        point2: tuple | None = None,
        center_point: tuple | None = None,
        radius: int | None = None,
        a: int | None = None,
        b: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    params_type = dict(
        point1=tuple,
        point2=tuple,
        center_point=tuple,
        radius=int,
        a=int,
        b=int
    )

    if mode == 'line':
        mode = 1
        params = dict(point1=point1, point2=point2)
    elif mode == 'circle':
        mode = 2
        params = dict(center_point=center_point, radius=radius)
    elif mode == 'ellipse':
        mode = 3
        params = dict(center_point=center_point, a=a, b=b)
    else:
        raise RuntimeError('mode not supported use {line, circle or ellipse}')

    msg_bad_type = " need to be type of "
    msg_bad_size = " need to be size of 2"
    msg_negative = " cannot be negative"
    msg_none = " parameter is missing"

    for key, itm in params.items():
        if itm is None:
            raise ValueError(key + msg_none)
        elif not isinstance(itm, params_type.get(key)):
            raise TypeError(key + msg_bad_type + str(params_type.get(key)))
        elif isinstance(itm, tuple):
            if len(itm) != 2:
                raise ValueError(key + msg_bad_size)
            if any(p < 0 for p in itm):
                raise ValueError(key + msg_negative)
        elif itm < 0:
            raise ValueError(key + msg_negative)

    yx = c_pycv.draw(mode, **params)

    if yx is None:
        raise RuntimeError('Error in C draw')

    return tuple(np.squeeze(a) for a in np.hsplit(yx, 2))


########################################################################################################################
