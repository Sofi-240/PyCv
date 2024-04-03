import numpy as np
from pycv._lib.array_api.regulator import np_compliance
from pycv._lib._src import c_pycv

__all__ = [
    'integral_image'
]


########################################################################################################################

def integral_image(image: np.ndarray) -> np.ndarray:
    image = np_compliance(image, arg_name='image', _check_finite=True, _check_atleast_nd=1)
    return c_pycv.integral_image(image)