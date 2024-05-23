import numpy as np
from pycv.io import ImageLoader, DEFAULT_DATA_PATH, show_scale_space, show_pyramid
from pycv.transform import rotate
from pycv._lib._src_py.pycv_pyramids import Layer
from pycv._lib.array_api.dtypes import cast

########################################################################################################################

loader = ImageLoader(DEFAULT_DATA_PATH)
TEMPLET = loader.load('photographer', _color_fmt='RGB2GRAY')
TEMPLET = cast(TEMPLET, np.float64)
IMAGE = rotate(TEMPLET, 90)


########################################################################################################################

class Parameters:
    def __init__(self, upsample=2, n_octaves=8, n_scales=3, sigma_min=1.6, sigma_in=0.5):
        self.upsample = upsample
        self.n_octaves = n_octaves
        self.n_scales = n_scales
        self.sigma_min = sigma_min / upsample
        self.sigma_in = sigma_in
        self.delta_min = 1 / upsample
        self.init_sigma = None
        self.deltas = None
        self.scalespace = None
        self.gaussian_sigmas = None

    def make_scalespace(self):
        self.init_sigma = self.upsample * np.sqrt(self.sigma_min ** 2 - self.sigma_in ** 2)
        deltas = self.delta_min * np.power(2, np.arange(self.n_octaves), dtype=np.float64)

        space = np.power(2, np.arange(self.n_scales + 3) / self.n_scales) * self.sigma_min

        scalespace = deltas[:, np.newaxis] / deltas[0] * space[np.newaxis, :]
        gaussian_sigmas = np.sqrt(np.diff(scalespace * scalespace, axis=1)) / deltas[:, np.newaxis]

        self.deltas = deltas
        self.scalespace = scalespace
        self.gaussian_sigmas = gaussian_sigmas

    def set_number_of_octaves(self, image_shape):
        size_min = 16
        max_octaves = int(np.log2((min(image_shape) * self.upsample) / size_min) + 1)
        if max_octaves < self.n_octaves:
            self.n_octaves = max_octaves

########################################################################################################################


params = Parameters()
params.make_scalespace()
params.set_number_of_octaves(TEMPLET.shape)


########################################################################################################################

UP_LAYER = Layer(
    2,
    dtype=TEMPLET.dtype,
    scales=round(float(params.scalespace[0, 0]), 4),
    factors=float(params.upsample),
    order=1,
)

UP_LAYER.input_dims = TEMPLET.shape

DOWN_LAYER = Layer(
    2,
    dtype=TEMPLET.dtype,
    scales=[round(float(i), 4) for i in params.scalespace[0, 1:]],
    factors=0.5,
    order=0,
)

DOWN_LAYER.input_dims = UP_LAYER.output_dims


########################################################################################################################
