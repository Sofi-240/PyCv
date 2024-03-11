import numpy as np
from pycv._lib.decorator import registrate_decorator
from pycv._lib.array_api.dtypes import cast, get_dtype_info
from pycv._lib.array_api.regulator import np_compliance, check_finite

__all__ = [
    "_Coordinates",
    'color_dispatcher',
]


########################################################################################################################

class _Coordinates:
    RGB2GRAY = [0.2989, 0.5870, 0.1140]
    RGB2YUV = [[0.299, -0.147, 0.615], [0.578, -0.289, -0.515], [0.114, 0.436, -0.100]]
    YUV2RGB = [[1., 1., 1.], [0., -0.395, 2.032], [1.140, -0.581, 0.]]

    @classmethod
    def get_coordinates(cls, transform_: str):
        try:
            return getattr(cls, transform_.upper())
        except AttributeError:
            raise Exception(f'{transform_} method is not supported')


########################################################################################################################

@registrate_decorator(kw_syntax=True)
def color_dispatcher(func, n_channels: int = 3, as_float: bool = True, same_type: bool = True, *args, **kwargs):
    image = np_compliance(args[0], arg_name='Image')
    check_finite(image, raise_err=True)

    if image.shape[-1] != n_channels:
        raise ValueError(
            f'Image need to have size {n_channels} along the last dimension got {image.shape}'
        )

    if not as_float:
        return func(image, *args[1:], **kwargs)

    dt = get_dtype_info(image.dtype)
    if dt.kind == 'f':
        float_image = image
    else:
        float_image = cast(image, np.float64)

    if not same_type:
        return func(float_image, *args[1:], **kwargs)

    out = func(float_image, *args[1:], **kwargs)
    return cast(out, dt.type)

########################################################################################################################
