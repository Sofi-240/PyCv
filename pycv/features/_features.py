import numpy as np
from pycv._lib.array_api.regulator import np_compliance
from .._lib.misc.calculations import derivatives
from pycv._lib._src import c_pycv

__all__ = [
    'integral_image',
    'structure_tensor',
]


########################################################################################################################

def integral_image(image: np.ndarray) -> np.ndarray:
    image = np_compliance(image, arg_name='image', _check_finite=True, _check_atleast_nd=1)
    return c_pycv.integral_image(image)


########################################################################################################################

def structure_tensor(
        inputs: np.ndarray,
        sigma: float | tuple = 1.,
        padding_mode: str = 'constant',
        constant_value: float = 0.,
        mode_xy: bool = False
) -> np.ndarray:
    return derivatives(inputs, sigma=sigma, padding_mode=padding_mode, constant_value=constant_value, mode_xy=mode_xy)


########################################################################################################################
