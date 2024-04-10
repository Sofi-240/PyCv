import numpy as np
from .._lib._src_py.pycv_transform import linear_interp1D
from ..measurements import histogram
from .._lib.array_api.dtypes import get_dtype_limits, cast
from .._lib._members_struct import Members

__all__ = [
    'adjust_gamma',
    'adjust_log',
    'adjust_exp',
    'adjust_linear',
    'hist_equalize'
]


########################################################################################################################

class Range(Members):
    IMAGE = 'image'
    DTYPE = 'dtype'


def _get_values_range(
        values: np.ndarray,
        vrange: Range | str | np.dtype | tuple[float, float],
        clip_negative: bool = False
) -> tuple[float, float]:

    if isinstance(vrange, tuple):
        vmin, vmax = map(float, vrange)
    elif isinstance(vrange, np.dtype):
        vmin, vmax = map(float, get_dtype_limits(vrange, include_negative=clip_negative))
    elif vrange == Range.DTYPE:
        vmin, vmax = map(float, get_dtype_limits(values.dtype, include_negative=clip_negative))
    else:
        vmin, vmax = map(float, (np.min(values), np.max(values)))

    if clip_negative:
        vmin = max(vmin, 0)

    return vmin, vmax


def _rescale_values(
        values: np.ndarray,
        out_range: Range | str | np.dtype | tuple[float, float] = Range.DTYPE,
        in_range: Range | str | np.dtype | tuple[float, float] = Range.IMAGE,
        clip_negative: bool = False
) -> np.ndarray:
    out_range = _get_values_range(values, out_range, clip_negative=clip_negative)
    in_range = _get_values_range(values, in_range, clip_negative=clip_negative)

    dvi = in_range[1] - in_range[0]
    dvo = out_range[1] - out_range[0]

    out = np.clip(values, in_range[0], in_range[1])

    if dvi == 0:
        return np.clip(out, out_range[0], out_range[1])

    out = (out - in_range[0]) * (dvo / dvi) + out_range[0]
    return out

########################################################################################################################


def adjust_gamma(image: np.ndarray, gamma: float = 1, gain: float = 1) -> np.ndarray:
    image = np.asarray_chkfinite(image)
    if np.max(image) < 0:
        raise ValueError('images must have non-negative values')
    if gamma < 0:
        raise ValueError('gamma must be non-negative value')
    dt = image.dtype
    if dt != 'uint8':
        v = _rescale_values(image, out_range=(0, 1), clip_negative=True)
        v = (gain * (v ** gamma)).astype(dt)
        return v

    v = 255 * gain * (np.linspace(0, 1, 256) ** gamma)
    v = np.clip(v, np.min(v), 255).astype(dt)

    return v[image]


def adjust_log(image: np.ndarray, gain: float = 1) -> np.ndarray:
    image = np.asarray_chkfinite(image)
    if np.max(image) < 0:
        raise ValueError('images must have non-negative values')

    v = _rescale_values(image, out_range=(0, 1), clip_negative=True)

    vt = gain * np.log2(v + 1)
    vt = _rescale_values(vt, out_range=_get_values_range(image, Range.IMAGE))
    vt = vt.astype(image.dtype)
    return vt


def adjust_exp(image: np.ndarray, gain: float = 1) -> np.ndarray:
    image = np.asarray_chkfinite(image)
    if np.max(image) < 0:
        raise ValueError('images must have non-negative values')
    v = _rescale_values(image, out_range=(0, 1), clip_negative=True)

    vt = np.exp(gain * v) - 1
    vt = _rescale_values(vt, out_range=_get_values_range(image, Range.IMAGE))
    vt = vt.astype(image.dtype)
    return vt


def adjust_linear(
        image: np.ndarray,
        out_range: str | tuple[float, float] = Range.DTYPE,
        in_range: str | tuple[float, float] = Range.IMAGE,
        clip_negative: bool = False
) -> np.ndarray:
    image = np.asarray_chkfinite(image)
    return _rescale_values(image, out_range, in_range, clip_negative)


########################################################################################################################

def hist_equalize(image: np.ndarray, n_bins: int = 256) -> np.ndarray:
    image = np.asarray_chkfinite(image)

    hist = histogram(image, n_bins)

    xn = image.ravel()
    xp = hist.bins
    fp = hist.cdf()[0]

    """
    why linear interpolation:
    _________________________
    
    Xp:  | 10 | 15 | 20 | 25 | ....
    _____|____|____|____|____|_____
    Np:  | 50 | 60 | 70 | 70 | ....
    _____|____|____|____|____|_____
    
         x0  xi              x1
    xp: -|---|---|---|---|---|----->
         10  11  12  13  14  15
    np: -|---|---|---|---|---|----->
         50  52  54  56  58  60
    
    Fp = Np / Np[-1]
    
            (xp[x1] - xp[xi]) x Np[x0] + (xp[xi] - xp[x0]) x Np[x1]
    X_out = _______________________________________________________   
                         (xp[xi] - xp[x0]) x Np[-1]
            
    
    #####################################################################################
    
            (15 - 11) x 50 + (11 - 10) x 60
    X_out = _______________________________ =  52
                         (15 - 10)
    
    Fp(X_out) = 52 / Np[-1]
    
    #####################################################################################
    """

    fn = linear_interp1D(xn, xp, fp)

    v_out = cast(fn.reshape(image.shape), image.dtype)

    return v_out


########################################################################################################################
