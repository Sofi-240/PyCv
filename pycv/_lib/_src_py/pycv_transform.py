import numpy as np
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib.array_api.dtypes import cast, get_dtype_info
from pycv._lib._src_py.utils import get_output, ctype_border_mode, ctype_interpolation_order, axis_transpose_to_last
from pycv._lib._src import c_pycv
from pycv._lib.filters_support.windows import gaussian_kernel
from pycv._lib._src_py.pycv_filters import convolve
from pycv._lib._src_py._geometric_transform import ProjectiveTransform,  _valid_matrix

__all__ = [
    'resize',
    'rotate',
    'geometric_transform'
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
    inputs = cast(inputs, np.float64)

    output, _ = get_output(None, inputs, output_shape)

    mode = ctype_border_mode(padding_mode)

    if anti_alias_filter is None:
        anti_alias_filter = order > 0 and any(so < si for so, si in zip(output_shape, inputs.shape))

    scale_factor = np.array([(si - 1) / (so - 1) for so, si in zip(output_shape, inputs.shape)])

    if anti_alias_filter:
        sigma = sigma is sigma if not None else np.max(scale_factor)
        kernel = gaussian_kernel(sigma, inputs.ndim)
        inputs = convolve(inputs, kernel, padding_mode=padding_mode, constant_value=constant_value)

    c_pycv.resize(inputs, output, order, 1, mode, constant_value)

    if preserve_dtype:
        return cast(output, dtype.type)

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

    angle = np.deg2rad(angle)
    c, s = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([[c, -s], [s, c]], dtype=np.float64)

    inputs_shape = np.asarray(inputs.shape[-2:])[::-1]

    center = (inputs_shape - 1) / 2

    if reshape:
        x, y = inputs_shape
        corners = np.asarray([[0, 0, x, x],
                              [0, y, 0, y]])
        bounds = rot_matrix @ corners
        output_shape = (bounds.ptp(axis=1) + 0.5).astype(int)
    else:
        output_shape = inputs_shape

    output_center = rot_matrix @ ((output_shape - 1) / 2)

    shift = center - output_center

    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = rot_matrix
    matrix[:2, 2] = shift

    dtype = get_dtype_info(inputs.dtype)
    inputs = cast(inputs, np.float64)

    output, _ = get_output(None, inputs, inputs.shape[:-2] + tuple(output_shape[::-1]))

    mode = ctype_border_mode(padding_mode)

    c_pycv.geometric_transform(matrix, inputs, output, None, None, order, mode, constant_value)

    if need_transpose:
        output = output.transpose(transpose_back)

    if preserve_dtype:
        return cast(output, dtype.type)

    return output


########################################################################################################################

def geometric_transform(
        inputs: np.ndarray,
        transform_matrix: ProjectiveTransform | np.ndarray,
        order: int = 1,
        axis: tuple | None = None,
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
    inputs = cast(inputs, np.float64)

    output, _ = get_output(None, inputs)

    mode = ctype_border_mode(padding_mode)

    c_pycv.geometric_transform(matrix, inputs, output, None, None, order, mode, constant_value)

    if need_transpose:
        output = output.transpose(transpose_back)

    if preserve_dtype:
        return cast(output, dtype.type)

    return output

########################################################################################################################
