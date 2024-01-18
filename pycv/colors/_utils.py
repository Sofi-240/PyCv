import numpy as np
from pycv._lib.decorator import registrate_decorator
from pycv._lib.array_api.dtypes import cast, get_dtype_info
from pycv._lib.array_api.regulator import np_compliance, check_finite

__all__ = [
    'color_dispatcher',
]


########################################################################################################################

@registrate_decorator(kw_syntax=True)
def color_dispatcher(
        func,
        n_channels: int = 3,
        same_type: bool = True,
        *args, **kwargs
):
    image = np_compliance(args[0], arg_name='Image')
    check_finite(image, raise_err=True)

    if image.shape[-1] != n_channels:
        raise ValueError(
            f'Image need to have size {n_channels} along the last dimension got {image.shape}'
        )

    dt = get_dtype_info(image.dtype)
    if dt.kind == 'f':
        float_image = image
    else:
        float_image = cast(image, np.float64)

    if not same_type:
        return func(float_image, *args, **kwargs)

    out = func(float_image, *args[1:], **kwargs)
    return cast(out, dt.type)

########################################################################################################################
