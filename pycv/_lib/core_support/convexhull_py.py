import numpy as np
from pycv._lib.core_support.utils import ctype_convex_hull_mode, axis_transpose_to_last
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib.core import ops

__all__ = [
    "convex_hull_2d"
]


########################################################################################################################

def convex_hull_2d(
        image: np.ndarray | None = None,
        points: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        convex_image: bool = True,
        image_shape: tuple | None = None,
        axis: tuple | None = None,
        mode: str = 'graham'
) -> tuple[np.ndarray] | tuple[np.ndarray, np.ndarray]:
    image_in = 1
    if image is None and points is None:
        raise ValueError('one of the parameters (image or points) need to be given')

    if image is not None:
        image = np.asarray(image, order='C')
        image = np_compliance(image, 'image', _check_finite=True)
        if mask is not None:
            mask = np.asarray(mask, order='C')
            if mask.shape != image.shape:
                raise ValueError('mask shape need to be same as image shape')
        nd = image.ndim
    else:
        image_in = 0
        points = np.asarray(points, order='C')
        points = np_compliance(points, 'points', _check_finite=True)

        if points.ndim != 2:
            raise ValueError('points need to have 2 dimensions (N points, nd)')

        nd = points.shape[1]

    output = None
    if convex_image:
        if image_in:
            image_shape = np.array(image.shape, np.int64)
        elif image_shape is None:
            image_shape = np.max(points, axis=0) + 1
        elif not (len(image_shape) == nd and all(image_shape[a] >= np.max(points[:, a]) + 1 for a in range(nd))):
            raise ValueError('image shape is smaller then the points maximum')

        output = np.zeros(image_shape, np.uint8)

    mode = ctype_convex_hull_mode(mode)

    if axis is not None and len(axis) != 2:
        raise ValueError('axes should contain exactly two values')

    if axis is None:
        axis = (nd - 2, nd - 1)

    need_transpose, transpose_forward, transpose_back = axis_transpose_to_last(nd, axis, default_nd=2)
    if image_in and need_transpose:
        image = image.transpose(transpose_forward)
        output = output.transpose(transpose_forward) if convex_image else output
    elif need_transpose:
        points[:, transpose_forward] = points[:, tuple(range(nd))]
        output = output.transpose(transpose_forward) if convex_image else output

    iter_ = [0] * (nd - 2)
    stop_ = [0] * (nd - 2)
    size = 1

    convex_points = []

    for i in range(nd - 2):
        stop_[i] = image.shape[i] - 1 if image_in else (np.max(points[:, i]) if not convex_image else output.shape[i] - 1)
        size *= (stop_[i] + 1)

    for i in range(size):
        p, o, ma, im = points, output, mask, image
        for j in range(nd - 2):
            im = im[iter_[j]] if image_in else None
            ma = ma[iter_[j]] if image_in and ma is not None else None
            p = p[p[:, j] == iter_[j], :] if not image_in else None
            o = o[iter_[j]] if convex_image else None

        p = p[:, -2:] if not image_in else None
        cp = ops.convex_hull(mode, im, ma, p, o)

        if cp is None:
            cp = np.ones((0, 2), np.int64)

        if nd > 2:
            one = np.ones((cp.shape[0], 1), cp.dtype)
            cp = np.hstack(tuple(one * iter_[j] for j in range(nd - 2)) + (cp, ))
            if need_transpose:
                cp[:, transpose_back] = cp[:, tuple(range(nd))]
        convex_points.append(cp)

        for j in range(nd - 3, -1, -1):
            if iter_[j] < stop_[j]:
                iter_[j] += 1
                break
            else:
                iter_[j] = 0

    if image_in and need_transpose and convex_image:
        output = output.transpose(transpose_back)
    elif need_transpose:
        points[:, transpose_back] = points[:, tuple(range(nd))]
        output = output.transpose(transpose_back) if convex_image else output

    if size > 1:
        convex_points = np.vstack(convex_points)
    else:
        convex_points = convex_points[0]

    if not convex_image:
        return convex_points

    return convex_points, output


########################################################################################################################

