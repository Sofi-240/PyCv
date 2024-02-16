import numpy as np
from pycv._lib._src_py.utils import ctype_convex_hull_mode, axis_transpose_to_last, valid_axis
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib.array_api import iterators
from pycv._lib._src import c_pycv

__all__ = [
    "convex_hull_2d",
    "convex_hull_2d_image"
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

    mode = ctype_convex_hull_mode(mode)

    if axis is not None and len(axis) != 2:
        raise ValueError('axes should contain exactly two values')
    if axis is None:
        axis = (nd - 2, nd - 1)

    need_transpose, transpose_forward, transpose_back = axis_transpose_to_last(nd, axis, default_nd=2)

    if need_transpose:
        if image_in:
            image = image.transpose(transpose_forward)
        else:
            points[:, transpose_forward] = points[:, tuple(range(nd))]

    if image_in:
        image_shape = np.array(image.shape, np.int64)
    elif image_shape is None:
        image_shape = np.max(points, axis=0) + 1
    elif convex_image and not (
            len(image_shape) == nd and all(image_shape[a] >= np.max(points[:, a]) + 1 for a in range(nd))):
        raise ValueError('image shape is smaller then the points maximum')
    else:
        image_shape = np.array(image_shape, np.int64)

    output = np.zeros(image_shape, np.uint8) if convex_image else None

    if nd == 2:
        convex_points = c_pycv.convex_hull(mode, image, mask, points, output)
    else:
        mask_in = mask is not None

        if image_in:
            _iter = iterators.ArrayIteratorSlice(image_shape, 2)
        else:
            _iter = iterators.PointsIteratorSlice(points, 2)

        convex_points = []
        slc_p = None
        for slc_a in _iter:
            if not image_in:
                slc_p = slc_a
                slc_a = tuple(s for s in points[slc_p[0]][0, :-2])
            cp = c_pycv.convex_hull(
                mode,
                image[slc_a] if image_in else None,
                mask[slc_a] if image_in and mask_in else None,
                points[slc_p] if not image_in else None,
                output[slc_a] if convex_image else None
            )
            cs = np.zeros((cp.shape[0], len(slc_a)), np.int64)
            if not image_in:
                cs[...] = points[slc_p[0]][0, :-2]
            convex_points.extend(np.hstack((cs, cp)))

        convex_points = np.vstack(convex_points)

    if need_transpose and convex_image:
        output = output.transpose(transpose_back)

    if need_transpose:
        if ~image_in:
            points[:, transpose_back] = points[:, tuple(range(nd))]
        convex_points[:, transpose_back] = convex_points[:, tuple(range(nd))]

    if not convex_image:
        return convex_points

    return convex_points, output


def convex_hull_2d_image(
        convex_hull: np.ndarray,
        output_shape: tuple | None = None,
        axis: tuple | None = None
) -> np.ndarray:
    convex_hull = np.asarray(convex_hull, order='C')
    convex_hull = np_compliance(convex_hull, 'convex_hull', _check_finite=True)

    if convex_hull.ndim != 2:
        raise ValueError('convex_hull need to have 2 dimensions (N points, nd)')

    ndim = convex_hull.shape[1]

    if output_shape is None:
        output_shape = np.amax(convex_hull, axis=0) + 1

    convex_image = np.zeros(output_shape, np.uint8)

    if ndim == 2:
        c_pycv.convex_hull_image(convex_image, convex_hull)
    else:
        _iter = iterators.PointsIteratorSlice(convex_hull, 2)
        axis = valid_axis(ndim, axis, 2)
        slc_axis = tuple(set(range(ndim)) - set(axis))

        for slc_p in _iter:
            slc_a = tuple(int(convex_hull[slc_p[0]][0, s]) for s in slc_axis)
            c_pycv.convex_hull_image(convex_image[slc_a], convex_hull[slc_p[0]][:, axis])

    return convex_image

########################################################################################################################
