import numpy as np
import abc
from pycv._lib._src import c_pycv
from ..array_api.dtypes import cast
from ..array_api.regulator import np_compliance
from .._src_py.utils import valid_axis, fix_kernel_shape, as_sequence_by_axis
from .._src_py.utils import get_output, ctype_border_mode, ctype_interpolation_order
from ._windows import gaussian_kernel

__all__ = [
    "GaussianPyramid",
    "LaplacianPyramid",
    "GaussianScaleSpace",
    "DOGPyramid"
]


########################################################################################################################

class _Scaler:

    def __init__(
            self,
            base_shape: tuple,
            axis: tuple,
            n_layers: int = -1,
            factors: tuple | float = 2,
            order: int = 1,
            downscale: bool = True
    ):
        self.factors = as_sequence_by_axis(factors, axis, len(base_shape), 1.)
        if any(f == 0 for f in self.factors):
            raise ValueError('scale factor cannot be zero')
        self.order = ctype_interpolation_order(order)
        self._is_downscale = downscale
        if downscale:
            self.n_layers = self.default_downscale_n_layers(base_shape, self.factors, n_layers)
        elif n_layers == -1:
            raise ValueError('n_layers must be positive for upscale mode')
        else:
            self.n_layers = n_layers
        self._base_shape = base_shape

    def __repr__(self):
        return f"{self.__class__.__name__}: shapes={self.output_shapes}"

    def __call__(self, inputs: np.ndarray, mode: str = 'constant', cval: float = 0.0) -> np.ndarray:
        output_shape = self.next_shape(inputs.shape)
        if any(s == 0 for s in output_shape):
            raise RuntimeError('input shape is to small for scale factors')
        mode = ctype_border_mode(mode)

        output, _ = get_output(None, inputs, output_shape)
        c_pycv.resize(inputs, output, self.order, 1, mode, cval)

        return output

    @property
    def _factors(self):
        return (1 / f if self._is_downscale else f for f in self.factors)

    @staticmethod
    def default_downscale_n_layers(base_shape: tuple, factors: tuple, n_layers: int) -> int:
        nn = min(int(np.log2((s / (f * 16)) + 1)) for s, f in zip(base_shape, factors) if f != 1)
        if n_layers == -1:
            return nn
        return min(nn, n_layers)

    @property
    def output_shapes(self) -> list[tuple]:
        prev_shape = self._base_shape
        out = []
        nn = self.n_layers
        while nn:
            prev_shape = self.next_shape(prev_shape)
            out.append(prev_shape)
            nn -= 1
        return out

    def next_shape(self, shape: tuple):
        return tuple(int(s * f + 0.5) for s, f in zip(shape, self._factors))


class _BasePyramid(abc.ABC):
    def __init__(
            self,
            base: np.ndarray,
            axis: tuple | None = None,
            n_layers: int = -1,
            downscale: tuple | float = 2,
            down_order: int = 1,
            upscale: tuple | float | None = None,
            up_order: int = 0,
            preserve_dtype: bool = False,
            padding_mode: str = 'symmetric',
            constant_value: float = 0.0
    ):
        self.base = np_compliance(base, 'base image', _check_finite=True)
        self._ndim = base.ndim
        axis = valid_axis(self._ndim, axis, 2)

        if upscale is not None:
            self._upscale = _Scaler(
                base.shape, axis=axis, n_layers=1, factors=upscale, order=up_order, downscale=False
            )
            init_shape = self._upscale.output_shapes[0]
        else:
            init_shape = base.shape
            self._upscale = None

        self._downscale = _Scaler(
            init_shape, axis=axis, n_layers=n_layers, factors=downscale, order=down_order, downscale=True
        )

        _base_dtype = self.base.dtype
        need_cast = preserve_dtype
        cast_safe = True

        if _base_dtype.kind != 'f':
            self.base = cast(self.base, np.float64)
        elif _base_dtype.itemsize != 8:
            self.base = self.base.astype(np.float64)
            cast_safe = False
        else:
            need_cast = False

        def output_cast(inputs: np.ndarray) -> np.ndarray:
            if not need_cast:
                return inputs.copy()
            if cast_safe:
                return cast(inputs, _base_dtype)
            return inputs.astype(_base_dtype)

        self._cast = output_cast

        def _scale(inputs: np.ndarray, scale_up: bool = False):
            if scale_up and not self._upscale:
                raise RuntimeError('upscale is not initialized')
            elif scale_up:
                return self._upscale(inputs, mode=padding_mode, cval=constant_value)
            return self._downscale(inputs, mode=padding_mode, cval=constant_value)

        self._scale = _scale

        self._layer = 0
        self._prev = None

    def __repr__(self):
        return f"{self.__class__.__name__}: " \
               f"{(self._upscale.output_shapes if self._upscale else []) + self._downscale.output_shapes}"

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __next__(self):
        pass

    @property
    def ndim(self):
        return self._ndim


########################################################################################################################

class GaussianPyramid(_BasePyramid):
    """
    Gaussian pyramid implementation.

    Gaussian pyramid is a type of image pyramid used in image processing and computer vision.
    It is constructed by repeatedly applying Gaussian smoothing and downsampling to the input image.
    Each level of the pyramid is created by blurring and subsampling the previous level.

    Parameters:
        base (numpy.ndarray): The base image of the pyramid.
        sigma (tuple or float): The standard deviation of the Gaussian kernel used for smoothing.
            If a tuple, it specifies the standard deviation along each dimension of the image.
            If a scalar, the same standard deviation is used along all dimensions.
        axis (tuple or None): The axes along which to apply the Gaussian smoothing. If None, all axes are used.
        n_layers (int): The number of layers in the pyramid. If -1, all possible layers are generated.
        downscale (tuple or float): The downsampling factor along each axis. If a tuple, it specifies
            the downsampling factor along each dimension of the image. If a scalar, the same downsampling
            factor is used along all dimensions.
        down_order (int): The order of downsampling operation. 1 for bilinear interpolation, 2 for quadratic,
            3 for cubic, etc.
        upscale (tuple or float or None): The upsampling factor along each axis. If None, no upsampling is performed.
        up_order (int): The order of upsampling operation. 0 for nearest neighbor interpolation,
            1 for bilinear interpolation, 2 for quadratic, 3 for cubic, etc.
        preserve_dtype (bool): Whether to preserve the data type of the input image in the pyramid levels.
        padding_mode (str): The padding mode to use during convolution.
        constant_value (float): The constant value to use for padding if padding_mode is 'constant'.

    Attributes:
        n_layers (int): The number of layers in the pyramid.

    """
    def __init__(
            self,
            base: np.ndarray,
            sigma: tuple | float = 1.,
            axis: tuple | None = None,
            n_layers: int = -1,
            downscale: tuple | float = 2,
            down_order: int = 1,
            upscale: tuple | float | None = None,
            up_order: int = 0,
            preserve_dtype: bool = False,
            padding_mode: str = 'symmetric',
            constant_value: float = 0.0
    ):
        if isinstance(sigma, tuple) and (len(sigma) == 0 or len(sigma) > base.ndim):
            raise ValueError('sigma must be a scalar or tuple of scalars with '
                             'size in range of 1 - base.ndim')

        axis = valid_axis(base.ndim, axis, 2 if np.isscalar(sigma) else len(sigma))

        super().__init__(
            base,
            axis=axis,
            n_layers=n_layers,
            downscale=downscale,
            down_order=down_order,
            upscale=upscale,
            up_order=up_order,
            preserve_dtype=preserve_dtype,
            padding_mode=padding_mode,
            constant_value=constant_value
        )

        mode = ctype_border_mode(padding_mode)
        cval = constant_value

        if np.isscalar(sigma) or (isinstance(sigma, tuple) and all(s == sigma[0] for s in sigma[1:])):
            kernel = gaussian_kernel(sigma if np.isscalar(sigma) else sigma[0], len(axis))
            kernel = np.reshape(kernel, fix_kernel_shape(kernel.shape, axis, self.ndim))
            offset = tuple(s // 2 for s in kernel.shape)

            def preprocess(inputs: np.ndarray):
                output = np.zeros_like(inputs)
                c_pycv.convolve(inputs, kernel, output, offset, mode, cval)
                return output

        else:
            kernels = []
            for s, ax in zip(sigma, axis):
                kernel = gaussian_kernel(s, 1)
                kernel = np.reshape(kernel, tuple(kernel.size if i == ax else 1 for i in range(self.ndim)))
                kernels.append(kernel)

            def preprocess(inputs: np.ndarray):
                output = inputs.copy()
                for k in kernels:
                    of = tuple(s // 2 for s in k.shape)
                    c_pycv.convolve(output.copy(), k, output, of, mode, cval)
                return output

        self._preprocess = preprocess

    def __len__(self):
        return self.n_layers

    def __iter__(self):
        self._layer = 0
        if self._upscale:
            inputs = self._preprocess(self.base)
            self._prev = self._scale(inputs, True)
        else:
            self._prev = self.base
        return self

    def __next__(self):
        if self._layer == self.n_layers:
            self._layer = 0
            self._prev = None
            raise StopIteration
        output = self._prev
        self._layer += 1
        inputs = self._preprocess(output)
        self._prev = self._scale(inputs, False)
        return self._cast(output)

    @property
    def n_layers(self) -> int:
        return self._downscale.n_layers + 1


########################################################################################################################

class LaplacianPyramid(GaussianPyramid):
    """
    Laplacian pyramid implementation.

    Laplacian pyramid is a type of image pyramid used in image processing and computer vision.
    It is constructed from a Gaussian pyramid by taking the difference between each level and its
    downsampled and blurred version. The resulting pyramid contains the details at each level.

    Parameters:
        base (numpy.ndarray): The base image of the pyramid.
        sigma (tuple or float): The standard deviation of the Gaussian kernel used for smoothing.
            If a tuple, it specifies the standard deviation along each dimension of the image.
            If a scalar, the same standard deviation is used along all dimensions.
        axis (tuple or None): The axes along which to apply the Gaussian smoothing. If None, all axes are used.
        n_layers (int): The number of layers in the pyramid. If -1, all possible layers are generated.
        downscale (tuple or float): The downsampling factor along each axis. If a tuple, it specifies
            the downsampling factor along each dimension of the image. If a scalar, the same downsampling
            factor is used along all dimensions.
        down_order (int): The order of downsampling operation. 1 for bilinear interpolation, 2 for quadratic,
            3 for cubic, etc.
        upscale (tuple or float or None): The upsampling factor along each axis. If None, no upsampling is performed.
        up_order (int): The order of upsampling operation. 0 for nearest neighbor interpolation,
            1 for bilinear interpolation, 2 for quadratic, 3 for cubic, etc.
        preserve_dtype (bool): Whether to preserve the data type of the input image in the pyramid levels.
        padding_mode (str): The padding mode to use during convolution.
        constant_value (float): The constant value to use for padding if padding_mode is 'constant'.

    Attributes:
        n_layers (int): The number of layers in the pyramid.

    Notes:
        - This class inherits from GaussianPyramid, which implements the Gaussian pyramid.
        - The Laplacian pyramid is constructed by taking the difference between each level of the Gaussian pyramid
          and its downsampled and smoothed version. It represents the details of the image at different scales.
    """

    def __iter__(self):
        self._layer = 0
        if self._upscale:
            inputs = self._scale(self.base, True)
            self._prev = (self._preprocess(inputs), inputs)
        else:
            self._prev = (self._preprocess(self.base), self.base)
        return self

    def __next__(self):
        if self._layer == self.n_layers:
            self._layer = 0
            self._prev = None
            raise StopIteration
        prev_prep, prev = self._prev
        self._layer += 1
        inputs = self._scale(prev_prep, False)
        self._prev = (self._preprocess(inputs), inputs)
        return self._cast(prev - prev_prep)


########################################################################################################################

class GaussianScaleSpace(_BasePyramid):
    """
    Gaussian Scale Space implementation.

    Gaussian scale space is a multi-scale representation of an image computed by convolving the image
    with Gaussian kernels of increasing standard deviations.

    Parameters:
        base (numpy.ndarray): The base image of the scale space.
        sigma (float): The standard deviation of the first Gaussian kernel.
        sigma_in (float): The standard deviation of the inner Gaussian kernels.
        axis (tuple or None): The axes along which to apply the Gaussian smoothing. If None, all axes are used.
        n_octaves (int): The number of octaves in the scale space.
        n_scales (int): The number of scales per octave.
        down_order (int): The order of downsampling operation. 0 for nearest neighbor interpolation,
            1 for bilinear interpolation, 2 for quadratic, 3 for cubic, etc.
        up_order (int): The order of upsampling operation. 0 for nearest neighbor interpolation,
            1 for bilinear interpolation, 2 for quadratic, 3 for cubic, etc.
        preserve_dtype (bool): Whether to preserve the data type of the input image in the scale space.
        padding_mode (str): The padding mode to use during convolution.
        constant_value (float): The constant value to use for padding if padding_mode is 'constant'.

    Attributes:
        n_octaves (int): The number of octaves in the scale space.

    Notes:
        - This class inherits from _BasePyramid, which implements the base functionality for image pyramids.
        - The Gaussian scale space is computed by convolving the image with Gaussian kernels of increasing
          standard deviations. The resulting images are downsampled to form a multi-scale representation
          of the original image.
    """
    def __init__(
            self,
            base: np.ndarray,
            sigma: float = 1.6,
            sigma_in: float = 0.5,
            axis: tuple | None = None,
            n_octaves: int = 8,
            n_scales: int = 3,
            down_order: int = 0,
            up_order: int = 1,
            preserve_dtype: bool = False,
            padding_mode: str = 'symmetric',
            constant_value: float = 0.0
    ):
        axis = valid_axis(base.ndim, axis, 2)

        super(GaussianScaleSpace, self).__init__(
            base,
            axis=axis,
            n_layers=n_octaves,
            downscale=2.,
            down_order=down_order,
            upscale=2.,
            up_order=up_order,
            preserve_dtype=preserve_dtype,
            padding_mode=padding_mode,
            constant_value=constant_value
        )

        self.n_scales = n_scales
        sigma = 0.5 * sigma
        _init_sigma = 2. * np.sqrt(sigma ** 2 - sigma_in ** 2)

        self.scalespace, self.gaussian_sigmas = self._init_scalespace(sigma)

        mode = ctype_border_mode(padding_mode)
        cval = constant_value

        kernels = []

        kernel = gaussian_kernel(_init_sigma, len(axis))
        kernel = np.reshape(kernel, fix_kernel_shape(kernel.shape, axis, self.ndim))
        kernels.append(kernel)

        for i in range(self.gaussian_sigmas.shape[1]):
            sigma = self.gaussian_sigmas[0, i]
            kernel = gaussian_kernel(sigma, len(axis))
            kernel = np.reshape(kernel, fix_kernel_shape(kernel.shape, axis, self.ndim))
            kernels.append(kernel)

        def preprocess_base(inputs: np.ndarray):
            output = np.zeros_like(inputs)
            c_pycv.convolve(inputs, kernels[0], output, tuple(s // 2 for s in kernels[0].shape), mode, cval)
            return output

        def preprocess(inputs: np.ndarray):
            output = np.zeros((self.n_scales + 3, *inputs.shape), inputs.dtype)
            output[0] = inputs
            for s, k in enumerate(kernels[1:]):
                of = tuple(s // 2 for s in k.shape)
                c_pycv.convolve(output[s], k, output[s + 1], of, mode, cval)
            return output

        self._preprocess_base = preprocess_base
        self._preprocess = preprocess

    def _init_scalespace(self, sigma: float) -> (np.ndarray, np.ndarray):
        deltas = 0.5 * np.power(2, np.arange(self.n_octaves), dtype=np.float64)
        space = np.power(2, np.arange(self.n_scales + 3) / self.n_scales) * sigma
        scalespace = deltas[:, np.newaxis] / deltas[0] * space[np.newaxis, :]
        gaussian_sigmas = np.sqrt(np.diff(scalespace * scalespace, axis=1)) / deltas[:, np.newaxis]
        return scalespace, gaussian_sigmas

    @property
    def n_octaves(self) -> int:
        return self._downscale.n_layers + 1

    def __iter__(self):
        self._octave = 0
        inputs = self._scale(self.base, True)
        self._prev = self._preprocess_base(inputs)
        return self

    def __next__(self):
        if self._octave == self.n_octaves:
            self._octave = 0
            self._prev = None
            raise StopIteration
        self._octave += 1
        output = self._preprocess(self._prev)
        self._prev = self._scale(output[self.n_scales], False)
        return self._cast(output)


class DOGPyramid(GaussianScaleSpace):
    """
    Difference of Gaussians (DoG) Pyramid implementation.

    DoG pyramid is a multi-scale representation of an image computed by taking the difference between
    consecutive scales in a Gaussian scale space.

    Parameters:
        base (numpy.ndarray): The base image of the scale space.
        sigma (float): The standard deviation of the first Gaussian kernel.
        sigma_in (float): The standard deviation of the inner Gaussian kernels.
        axis (tuple or None): The axes along which to apply the Gaussian smoothing. If None, all axes are used.
        n_octaves (int): The number of octaves in the scale space.
        n_scales (int): The number of scales per octave.
        down_order (int): The order of downsampling operation. 0 for nearest neighbor interpolation,
            1 for bilinear interpolation, 2 for quadratic, 3 for cubic, etc.
        up_order (int): The order of upsampling operation. 0 for nearest neighbor interpolation,
            1 for bilinear interpolation, 2 for quadratic, 3 for cubic, etc.
        preserve_dtype (bool): Whether to preserve the data type of the input image in the scale space.
        padding_mode (str): The padding mode to use during convolution.
        constant_value (float): The constant value to use for padding if padding_mode is 'constant'.

    Notes:
        - This class inherits from GaussianScaleSpace, which computes the Gaussian scale space
          representation of an image.
        - The DoG pyramid is computed by taking the difference between consecutive scales in a Gaussian
          scale space. It provides a multi-scale representation of the image with enhanced features.

    Attributes:
        n_octaves (int): The number of octaves in the scale space.

    Example:
        # Create a DoG pyramid from an image
        image = np.random.rand(256, 256)
        dog_pyramid = DOGPyramid(image, sigma=1.6, sigma_in=0.5, n_octaves=8, n_scales=3)

        # Iterate over the pyramid and access each octave
        for octave in dog_pyramid:
            # Process each octave of the DoG pyramid
            process_octave(octave)

    """
    def __next__(self):
        output = super().__next__()
        return np.diff(output, axis=0)