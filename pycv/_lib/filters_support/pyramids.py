import numpy as np
import abc
from pycv._lib._src.c_pycv import CLayer
from ..array_api.dtypes import cast
from ..array_api.regulator import np_compliance
from .._src_py.utils import valid_axis, as_sequence_by_axis
from .._src_py.utils import ctype_border_mode, ctype_interpolation_order

__all__ = [
    "GaussianPyramid",
    "LaplacianPyramid",
    "GaussianScaleSpace",
    "DOGPyramid"
]


########################################################################################################################

def _make_scalespace(ndim: int, scales: list | tuple | float = None) -> list[tuple]:
    def scalar_valid(_v):
        if _v < 0:
            raise ValueError('scale cannot be negative')
        return True

    def tuple_valid(_v):
        if len(_v) != ndim:
            raise ValueError('scale tuple size must be equal to ndim')
        return all(filter(scalar_valid, _v))

    if scales is None:
        return

    if np.isscalar(scales) and scalar_valid(scales):
        return [(scales,) * ndim]
    elif isinstance(scales, tuple) and tuple_valid(scales):
        return [scales]
    elif isinstance(scales, list) and all(filter(tuple_valid, scales)):
        return scales
    raise TypeError('invalid scales input')


def _factor_valid(ndim: int, factors: tuple | float) -> tuple:
    if factors is None:
        return

    def scalar_valid(_v):
        if _v <= 0:
            raise ValueError('factor cannot be negative or zero')
        return True

    if np.isscalar(factors) and scalar_valid(factors):
        return (factors,) * ndim
    elif isinstance(factors, tuple):
        if len(factors) != ndim:
            raise ValueError('factors size must be equal to ndim')
        if all(filter(scalar_valid, factors)):
            return factors
    raise TypeError('invalid factors input')


class Layer(CLayer):
    def __init__(
            self,
            ndim: int,
            dtype: np.dtype | None = None,
            scales: list | tuple | float = None,
            order: int = 0,
            factors: tuple | float = None,
            padding_mode: str = 'symmetric',
            cval: float = 0.
    ):
        if ndim <= 0:
            raise TypeError("ndim must be positive integer")
        if dtype is not None:
            dtype = np.dtype(dtype)
        padding_mode = ctype_border_mode(padding_mode)
        order = ctype_interpolation_order(order)
        super().__init__(ndim, dtype=dtype, padding=padding_mode, order=order, cval=cval)
        if scales is not None:
            self.scalespace = _make_scalespace(ndim, scales)
        if factors is not None:
            self.factors = _factor_valid(ndim, factors)
        self.next = None

    @property
    def is_downscale(self) -> bool:
        return any(v < 1. for v in self.factors)

    @property
    def is_upscale(self) -> bool:
        return any(v > 1. for v in self.factors)

    @property
    def default_n_layers(self) -> int:
        if any(s == 0 for s in self.input_dims):
            return 0
        if not self.is_downscale:
            return 1
        nn = min(int(np.log2(((s * f) / 16) + 1)) for s, f in zip(self.input_dims, self.factors) if f != 1)
        return int(nn)

    def scale(self, inputs: np.ndarray):
        if not self.scalespace:
            return inputs.copy()
        return super().scale(inputs)


class PyramidBase(abc.ABC):
    def __init__(
            self,
            ndim: int,
            nlayers: int,
            scales: list | tuple | float = None,
            order: int = 0,
            factors: tuple | float = 0.5,
            preprocess_scales: list | tuple | float = None,
            preprocess_order: int = 1,
            preprocess_factors: tuple | float = None,
            padding_mode: str = 'symmetric',
            preserve_dtype: bool = False,
            cval: float = 0.
    ):
        if factors is None:
            factors = 0.5
        dtype = np.dtype(np.float64)
        self.layer = Layer(ndim, dtype, scales, order, factors, padding_mode, cval)
        if not self.layer.is_downscale:
            raise TypeError('invalid pyramid down factors')
        self.preprocess_layer = None

        if preprocess_factors is not None or preprocess_scales is not None:
            self.preprocess_layer = Layer(ndim, dtype, preprocess_scales, preprocess_order, preprocess_factors,
                                          padding_mode, cval)

        self._nlayers = nlayers
        self.preserve_dtype = preserve_dtype
        self._shape = None

        def cast_f(arr):
            return arr

        def cast_b(arr):
            return arr

        self._cast_input = cast_f
        self._cast_output = cast_b

    @property
    def ndim(self):
        return self.layer.ndim

    @property
    def input_shape(self):
        if self.preprocess_layer is not None:
            return self.preprocess_layer.input_dims
        return self._shape

    @input_shape.setter
    def input_shape(self, shape: tuple):
        if len(shape) != self.layer.ndim:
            raise ValueError('shape size must be equal to ndim')
        if any(s <= 0 for s in shape):
            raise ValueError('shape must have positive integers')
        if self.preprocess_layer is None:
            self._shape = shape
            self.layer.input_dims = shape
        else:
            self.preprocess_layer.input_dims = shape
            self.layer.input_dims = self.preprocess_layer.output_dims

    @property
    def default_nlayers(self) -> int:
        if any(d == 0 for d in self.layer.output_dims):
            return 0
        nn = 2 if self.preprocess_layer is None else 3
        if self.layer.is_downscale:
            nn += min(
                int(np.log2(((s * f) / 16) + 1)) for s, f in zip(self.layer.output_dims, self.layer.factors) if f != 1)
        return nn

    @property
    def nlayers(self):
        if self._nlayers == -1:
            return self.default_nlayers + 1
        return min(self.default_nlayers + 1, self._nlayers)

    @nlayers.setter
    def nlayers(self, n: int):
        self._nlayers = n

    def _prepare(self, inputs: np.ndarray):
        inputs = np_compliance(inputs, arg_name='inputs', _check_finite=True)
        if inputs.ndim != self.ndim:
            raise TypeError('invalid ndim for inputs')
        self.input_shape = inputs.shape
        dtype = inputs.dtype
        preserve_dtype = self.preserve_dtype

        if inputs.dtype.kind != "f":
            def cast_f(arr):
                return cast(arr, self.layer.input_dtype)

            def cast_b(arr):
                return cast(arr, dtype) if preserve_dtype else arr

        elif inputs.dtype.itemsize != 8:
            def cast_f(arr):
                return arr.astype(self.layer.input_dtype)

            def cast_b(arr):
                return arr.astype(dtype) if preserve_dtype else arr

        else:
            def cast_f(arr):
                return arr

            def cast_b(arr):
                return arr

        self._cast_input = cast_f
        self._cast_output = cast_b
        return inputs

    @abc.abstractmethod
    def __call__(self, inputs: np.ndarray):
        pass


########################################################################################################################

class GaussianPyramid(PyramidBase):

    def __init__(
            self,
            ndim: int,
            sigma: tuple | float = 1.,
            axis: tuple = None,
            n_layers: int = -1,
            factors: tuple | float = 0.5,
            order: int = 0,
            preserve_dtype: bool = False,
            padding_mode: str = 'symmetric',
            cval: float = 0.0
    ):
        axis = valid_axis(ndim, axis, ndim)
        sigma = as_sequence_by_axis(sigma, axis, ndim, 0.0)

        super().__init__(
            ndim, nlayers=n_layers, scales=sigma, factors=factors, order=order,
            preserve_dtype=preserve_dtype, padding_mode=padding_mode, cval=cval
        )

    def __call__(self, inputs: np.ndarray):
        inputs = self._prepare(inputs)

        output = self._cast_input(inputs)
        nn = self.nlayers - 1

        output = self.layer.scale(output)[-1]
        yield self._cast_output(output)

        while nn:
            output = self.layer.rescale(output)
            self.layer.reduce()
            output = self.layer.scale(output)[-1]
            nn -= 1
            yield self._cast_output(output)

        self.input_shape = inputs.shape


class LaplacianPyramid(GaussianPyramid):
    def __call__(self, inputs: np.ndarray):
        inputs = self._prepare(inputs)

        prev_output = self._cast_input(inputs)
        nn = self.nlayers - 1

        output = self.layer.scale(prev_output)[-1]
        yield self._cast_output(prev_output - output)

        while nn:
            prev_output = self.layer.rescale(output)
            self.layer.reduce()
            output = self.layer.scale(prev_output)[-1]
            nn -= 1
            yield self._cast_output(prev_output - output)

        self.input_shape = inputs.shape


########################################################################################################################

class GaussianScaleSpace(PyramidBase):
    def __init__(
            self,
            ndim: int,
            scalespace: list[tuple] | list[int],
            n_layers: int = -1,
            factors: tuple | float = 0.5,
            order: int = 0,
            rescale_index: int = -1,
            preprocess_scales: list[tuple] | tuple | float = None,
            preprocess_order: int = 1,
            preprocess_factors: tuple | float = None,
            preserve_dtype: bool = False,
            padding_mode: str = 'symmetric',
            cval: float = 0.0
    ):
        if not len(scalespace):
            raise ValueError('scalespace must be non empty')
        if np.isscalar(scalespace[0]):
            scalespace = [(s,) * ndim for s in scalespace]
        if rescale_index >= len(scalespace):
            raise ValueError('rescale_index is out of range')
        self.rescale_index = rescale_index
        super().__init__(
            ndim, nlayers=n_layers, scales=scalespace, factors=factors, order=order,
            preprocess_scales=preprocess_scales, preprocess_order=preprocess_order,
            preprocess_factors=preprocess_factors,
            preserve_dtype=preserve_dtype, padding_mode=padding_mode, cval=cval
        )

    def __call__(self, inputs: np.ndarray):
        inputs = self._prepare(inputs)

        output = self._cast_input(inputs)
        nn = self.nlayers - 1

        if self.preprocess_layer is not None:
            output = self.preprocess_layer.scale(output)
            output = self.preprocess_layer.rescale(output[self.rescale_index])
            nn -= 1

        output = self.layer.scale(output)
        yield self._cast_output(output)

        while nn:
            output = self.layer.rescale(output[self.rescale_index])
            self.layer.reduce()
            output = self.layer.scale(output)
            nn -= 1
            yield self._cast_output(output)

        self.input_shape = inputs.shape


class DOGPyramid(GaussianScaleSpace):
    def __call__(self, inputs: np.ndarray):
        inputs = self._prepare(inputs)

        output = self._cast_input(inputs)
        nn = self.nlayers

        if self.preprocess_layer is not None:
            output = self.preprocess_layer.scale(output)
            output = self.preprocess_layer.rescale(output[self.rescale_index])
            nn -= 1

        output = self.layer.scale(output)
        yield self._cast_output(np.diff(output, axis=0))

        while nn:
            output = self.layer.rescale(output[self.rescale_index])
            self.layer.reduce()
            output = self.layer.scale(output)
            nn -= 1
            yield self._cast_output(np.diff(output, axis=0))

        self.input_shape = inputs.shape

########################################################################################################################
