import numpy as np
from ..array_api.regulator import np_compliance
from ..array_api.dtypes import cast, get_dtype_info
from ..array_api.shapes import atleast_nd
from .._src_py.utils import get_output, ctype_border_mode, ctype_interpolation_order, axis_transpose_to_last, \
    ctype_hough_mode
from pycv._lib._src import c_pycv
from ..filters_support._windows import gaussian_kernel
from .._src_py.pycv_filters import convolve
from ._geometric_transform import ProjectiveTransform, SimilarityTransform, _valid_matrix

__all__ = [
    'resize',
    'rotate',
    'geometric_transform',
    'hough_transform',
    'linear_interp1D',
]


########################################################################################################################

def resize(
        inputs: np.ndarray,
        output_shape: tuple,
        order: int = 1,
        anti_alias_filter: bool | None = None,
        sigma: float | None = None,
        padding_mode: str = 'constant',
        constant_value: float | int | None = 0,
        preserve_dtype: bool = False
) -> np.ndarray:
    inputs = np_compliance(inputs, 'Input', _check_finite=True)

    if any(s == 0 for s in output_shape) or any(s == 0 for s in inputs.shape):
        raise ValueError('array shape cannot be zero')

    order = ctype_interpolation_order(order)

    dtype = get_dtype_info(inputs.dtype)
    if dtype.kind != 'f':
        inputs = cast(inputs, np.float64)
    elif dtype.itemsize != 8:
        inputs = inputs.astype(np.float64)
    else:
        preserve_dtype = False

    output, _ = get_output(None, inputs, output_shape)

    mode = ctype_border_mode(padding_mode)

    if anti_alias_filter is None:
        anti_alias_filter = order > 0 and any(so < si for so, si in zip(output_shape, inputs.shape))

    scale_factor = np.array([(si - 1) / (so - 1) for so, si in zip(output_shape, inputs.shape)])

    if anti_alias_filter:
        sigma = sigma if sigma is not None else np.maximum(0, np.max((scale_factor - 1) / 2))
        for i in range(inputs.ndim):
            if scale_factor[i] <= 1:
                continue
            kernel = gaussian_kernel(sigma, 1)
            kernel = kernel.reshape(tuple(1 if j != i else kernel.size for j in range(inputs.ndim)))
            inputs = convolve(inputs, kernel, padding_mode=padding_mode, constant_value=constant_value)

    c_pycv.resize(inputs, output, order, 1, mode, constant_value)

    if preserve_dtype:
        min_val = np.min(inputs)
        max_val = np.max(inputs)
        np.clip(output, min_val, max_val, out=output)
        if dtype.kind != 'f':
            cast(output, dtype.type)
        elif dtype.itemsize != 8:
            output = cast(output, dtype.type)
        return output

    return output


########################################################################################################################

def rotate(
        inputs: np.ndarray,
        angle: float,
        order: int = 1,
        axis: tuple | None = None,
        reshape: bool = True,
        padding_mode: str = 'constant',
        constant_value: float | int | None = 0,
        preserve_dtype: bool = False
) -> np.ndarray:
    inputs = np_compliance(inputs, 'Input', _check_finite=True)

    if axis is not None and len(axis) != 2:
        raise ValueError('axes should contain exactly two values')

    need_transpose, transpose_forward, transpose_back = axis_transpose_to_last(inputs.ndim, axis, default_nd=2)

    if need_transpose:
        inputs = inputs.transpose(transpose_forward)

    if any(s == 0 for s in inputs.shape):
        raise ValueError('array shape cannot be zero')

    order = ctype_interpolation_order(order)

    inputs_shape = np.asarray(inputs.shape[-2:])[::-1]
    center = (inputs_shape - 1) / 2

    matrix = SimilarityTransform(translation=center) @ SimilarityTransform(rotation=angle) @ SimilarityTransform(translation=-center)

    if reshape:
        x, y = inputs_shape
        corners = np.array([
            [0, 0],
            [0, x - 1],
            [y - 1, x - 1],
            [y - 1, 0]
        ])
        bounds = SimilarityTransform(matrix=matrix).inverse(corners)[..., :-1]
        output_shape = (bounds.ptp(axis=0) + 0.5).astype(int)

        matrix = matrix @  SimilarityTransform(translation=bounds.min(axis=0))
    else:
        output_shape = inputs_shape

    dtype = get_dtype_info(inputs.dtype)
    if dtype.kind != 'f':
        inputs = cast(inputs, np.float64)
    elif dtype.itemsize != 8:
        inputs = inputs.astype(np.float64)

    output, _ = get_output(None, inputs, inputs.shape[:-2] + tuple(output_shape[::-1]))

    mode = ctype_border_mode(padding_mode)

    c_pycv.geometric_transform(matrix.matrix, inputs, output, None, None, order, mode, constant_value)

    if need_transpose:
        output = output.transpose(transpose_back)

    if preserve_dtype:
        min_val = np.min(inputs)
        max_val = np.max(inputs)
        np.clip(output, min_val, max_val, out=output)
        if dtype.kind != 'f':
            cast(output, dtype.type)
        elif dtype.itemsize != 8:
            output = cast(output, dtype.type)
        return output

    return output


########################################################################################################################

def geometric_transform(
        inputs: np.ndarray,
        transform_matrix: ProjectiveTransform | np.ndarray,
        order: int = 1,
        axis: tuple | None = None,
        output_shape: tuple | None = None,
        padding_mode: str = 'constant',
        constant_value: float | int | None = 0,
        preserve_dtype: bool = False
) -> np.ndarray:
    inputs = np_compliance(inputs, 'Input', _check_finite=True)
    matrix = np_compliance(np.asarray(transform_matrix), 'Matrix', _check_finite=True)
    matrix = _valid_matrix(matrix.shape[0] - 1, matrix)

    t_ndim = matrix.shape[0] - 1

    if any(s == 0 for s in inputs.shape):
        raise ValueError('array shape cannot be zero')

    if axis is not None and len(axis) != t_ndim:
        raise ValueError(f'axes should contain exactly {t_ndim} values')

    need_transpose, transpose_forward, transpose_back = axis_transpose_to_last(inputs.ndim, axis, default_nd=t_ndim)

    if need_transpose:
        inputs = inputs.transpose(transpose_forward)

    order = ctype_interpolation_order(order)

    dtype = get_dtype_info(inputs.dtype)
    if dtype.kind != 'f':
        inputs = cast(inputs, np.float64)
    elif dtype.itemsize != 8:
        inputs = inputs.astype(np.float64)

    output, _ = get_output(None, inputs, output_shape)

    c_pycv.geometric_transform(matrix, inputs, output, None, None, order, ctype_border_mode(padding_mode), constant_value)

    if need_transpose:
        output = output.transpose(transpose_back)

    if preserve_dtype:
        min_val = np.min(inputs)
        max_val = np.max(inputs)
        np.clip(output, min_val, max_val, out=output)
        if dtype.kind != 'f':
            cast(output, dtype.type)
        elif dtype.itemsize != 8:
            output = cast(output, dtype.type)
        return output

    return output


########################################################################################################################


def hough_transform(
        hough_mode: str,
        inputs: np.ndarray,
        params: np.ndarray | None = None,
        offset: int | None = None,
        normalize: bool = False,
        expend: bool = False,
        threshold: int = 10,
        line_length: int = 50,
        line_gap: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
    inputs = np_compliance(inputs, 'Input', _check_finite=True)
    inputs = atleast_nd(inputs, 2, raise_err=True)

    dist = None
    if hough_mode in ['line', 'probabilistic_line', 'pp_line']:
        if params is not None:
            params = np_compliance(params, 'theta', _check_finite=True)
        else:
            params = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
        if offset is None:
            offset = int(np.ceil(np.hypot(inputs.shape[-2], inputs.shape[-1])))
            dist = np.linspace(-offset, offset, 2 * offset + 1)
        elif offset <= 0:
            raise ValueError('offset cannot be zero')
        else:
            dist = np.linspace(-offset, offset, 2 * offset + 1)
        if hough_mode == 'line':
            params_in = dict(offset=offset)
        else:
            params_in = dict(offset=offset, threshold=threshold, line_length=line_length, line_gap=line_gap)
    elif hough_mode == 'circle':
        if params is None:
            raise ValueError('radius must be given')
        elif np.isscalar(params):
            params = np.array([params], np.int64)
        params = np_compliance(params, 'radius', _check_finite=True)
        params_in = dict(normalize=normalize, expend=expend)
    else:
        raise ValueError('hough mode is not supported')

    output = c_pycv.hough_transform(ctype_hough_mode(hough_mode), inputs, params, **params_in)

    if hough_mode == 'line':
        return output, params, dist

    return output


########################################################################################################################

def linear_interp1D(
        xn: np.ndarray,
        xp: np.ndarray,
        fp: np.ndarray,
        l: float | None = None,
        r: float | None = None
) -> np.ndarray:
    xn = np_compliance(xn, 'xn', _check_finite=True)
    xp = np_compliance(xp, 'xp', _check_finite=True)
    fp = np_compliance(fp, 'fp', _check_finite=True)

    if any(x.ndim != 1 for x in (xn, xp, fp)):
        raise ValueError('all of the inputs (xn, xp, fp) must be 1d')
    if xp.shape != fp.shape:
        raise ValueError('xp and fp must have the same size')

    ind = np.argsort(xp, kind='mergesort')
    xp = xp[ind]
    fp = fp[ind]

    if l is None:
        l = float(fp[0])
    if r is None:
        r = float(fp[-1])

    return c_pycv.linear_interp1D(xn, xp, fp, l, r)

########################################################################################################################
