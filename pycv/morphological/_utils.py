import numpy as np
from pycv._lib.filters_support.kernel_utils import default_binary_strel
from pycv._lib.decorator import registrate_decorator
from pycv._lib.array_api.dtypes import as_binary_array
from pycv._lib.array_api.regulator import np_compliance, check_finite

__all__ = [
    'default_strel',
    'RAVEL_ORDER',
    'FLIPPER',
    'PUBLIC'
]

PUBLIC = []

RAVEL_ORDER = 'C'
FLIPPER = (1, 0, 2)
SUPPORTED_MODE = {
    'B',  # binary
    'G',  # gray
    'A'  # all
}
SUPPORTED_OPS = {'ERO', 'DIL'}


########################################################################################################################

def default_strel(
        ndim: int,
        strel: np.ndarray | None,
        connectivity=1,
        hole=False,
        flip: bool = True,
        dtype_bool: bool = False,
        offset: tuple | None = None
) -> np.ndarray:
    if strel is None:
        strel = default_binary_strel(ndim, connectivity)
    else:
        if not isinstance(strel, np.ndarray):
            raise TypeError(f'Strel need to be type of numpy.ndarray')

        if offset is not None:
            if not isinstance(offset, tuple):
                raise TypeError(f'offset point need to be type of tuple got {type(offset)}')
            if len(offset) != strel.ndim:
                raise ValueError(
                    f'Number of dimensions in center and kernel do not match: {len(offset)} != {strel.ndim}')
            if not all(of < s for of, s in zip(offset, strel.shape)):
                raise ValueError(f'offset point is out of range for Strel with shape of {strel.shape}')
        else:
            if not all(s % 2 != 0 for s in strel.shape):
                raise ValueError('Structuring element dimensions length need to be odd or set offset point')
            offset = (s // 2 for s in strel.shape)

        if dtype_bool and strel.dtype != bool:
            raise ValueError(f'strel dtype need to be boolean')

        if strel.ndim != ndim:
            raise ValueError(
                f'Number of dimensions in strel and img does not match {strel.ndim} != {ndim}'
            )
        if flip:
            strel = np.flip(strel, FLIPPER[:strel.ndim]) if strel.ndim > 1 else np.flip(strel, 0)
        if hole:
            strel = strel.copy()
            strel[offset] = False

    return strel


########################################################################################################################

@registrate_decorator(kw_syntax=True)
def morph_dispatcher(
        func,
        mode: str = 'B',
        strel_def: bool = True,
        output_def: bool = True,
        mask_def: bool = True,
        *args, **kwargs
):
    image = np_compliance(args[0], 'Image')
    if not check_finite(image):
        raise ValueError('Kernel must not contain infs or NaNs')

    ndim = image.ndim
    if ndim > 3:
        raise ValueError(f'Morphological operation on 4D or above is not supported, got image with rank of {ndim}')

    if mode == 'B':
        image = as_binary_array(image, 'Image')
    elif mode == 'G' and image.dtype == bool:
        raise ValueError(f'Gray level img cannot be with boolean dtype')

    if strel_def:
        strel = default_strel(
            image.ndim, kwargs.get('strel', None),
            connectivity=1, hole=False, flip=True, dtype_bool=mode == 'B',
            offset=kwargs.get('offset', None)
        )
        if strel.dtype != image.dtype:
            raise ValueError(f'strel dtype need to be boolean or as image dtype')

        if not all(map(lambda nk: nk[0] >= nk[1], zip(image.shape, strel.shape))):
            raise ValueError(f"Strel can't be bigger than input image in terms of shape")

        kwargs['strel'] = strel

    output = None
    if output_def:
        output = kwargs.get('output', None)
        if output is None:
            output = np.zeros_like(image, dtype=image.dtype)
        else:
            if not isinstance(output, np.ndarray):
                raise TypeError(f'Output need to be type of numpy.ndarray')

            if not (output.ndim == image.ndim and all(sa == sm for sa, sm in zip(output.shape, image.shape))):
                raise ValueError(f'Image and output shape does not match {image.shape} != {output.shape}')
            if image.dtype != output.dtype:
                output[:] = output.astype(image.dtype)

        kwargs['output'] = output

    if mask_def:
        mask = kwargs.get('mask', None)
        if mask is not None:
            if not isinstance(mask, np.ndarray):
                raise TypeError(f'mask need to be type of numpy.ndarray')

            if mask.dtype != bool:
                raise ValueError(f'mask need to have boolean dtype')

            if not (mask.ndim == image.ndim and all(sa == sm for sa, sm in zip(mask.shape, image.shape))):
                raise ValueError(f'img and mask shape does not match {image.shape} != {mask.shape}')

        kwargs['mask'] = mask

    if np.all(image == 0):
        if output is not None:
            output[(None,) * image.ndim] = 0
            return output
        return np.zeros_like(image, dtype=image.dtype)

    return func(image, *args[1:], **kwargs)


########################################################################################################################


def binary_dispatcher(func, *args, **kwargs):
    return morph_dispatcher(func, mode='B', strel_def=True, output_def=True, mask_def=True, *args, **kwargs)

########################################################################################################################


