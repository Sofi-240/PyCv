import numpy as np

__all__ = [
    'SUPPORTED_MODE',
    'valid_odd_kernel',
    'valid_pad_mode',
    'get_padding_width',
    'pad',
    'PUBLIC'
]

PUBLIC = []

# from numpy array pad supported keywords passed for the pad mode

SUPPORTED_MODE = {
    'valid': [],
    'edge': [],
    'wrap': [],
    'constant': ['constant_values'],
    'linear_ramp': ['end_values'],
    'maximum': ['stat_length'],
    'mean': ['stat_length'],
    'median': ['stat_length'],
    'minimum': ['stat_length'],
    'reflect': ['reflect_type'],
    'symmetric': ['reflect_type'],
    'same': []
}


########################################################################################################################

def valid_pad_mode(
        mode: str
) -> str:
    if mode == 'same':
        return 'constant'
    if mode not in SUPPORTED_MODE:
        raise ValueError(f'Padding mode {mode} id not supported')
    return mode


def clean_kw(
        mode: str,
        **kwargs
):
    al = SUPPORTED_MODE.get(mode, [])
    out = dict()
    for arg in al:
        a = kwargs.get(arg, None)
        if a is None: continue
        out[arg] = a
    return out


def valid_odd_kernel(
        kernel_shape: tuple
) -> None:
    if not all((s % 2) != 0 for s in kernel_shape):
        raise ValueError(f'Expected kernel dimensions length to be odd')


def get_padding_width(
        kernel_shape: tuple,
        offset: tuple | None = None,
        flip: bool = False,
        image_shape: tuple | None = None
) -> tuple:
    if offset is None:
        offset = tuple(s // 2 for s in kernel_shape)

    if len(kernel_shape) != len(offset):
        raise ValueError(
            f'Number of dimensions in kernel and offset does not match: {len(kernel_shape)} != {len(offset)}'
        )

    if image_shape is not None:
        if len(image_shape) < len(kernel_shape):
            raise ValueError(
                f'Image rank is smaller then the kernel rank'
            )

        pad_width = ((0, 0),) * (len(image_shape) - len(kernel_shape))
    else:
        pad_width = tuple()

    if not flip:
        pad_width += tuple((of, s - of - 1) for of, s in zip(offset, kernel_shape))
    else:
        pad_width += tuple((s - of - 1, of) for of, s in zip(offset, kernel_shape))

    return pad_width


def pad(
        inputs: np.ndarray,
        pad_width: tuple,
        mode: str = 'same',
        **kwargs
) -> np.ndarray:
    if mode == 'valid':
        return inputs.copy()
    mode = valid_pad_mode(mode)
    pad_kw = clean_kw(mode, **kwargs)
    return np.pad(inputs, pad_width=pad_width, mode=mode, **pad_kw)

########################################################################################################################
