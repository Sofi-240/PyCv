import numpy as np
from .._lib.array_api.regulator import np_compliance
from .._lib._src_py import pycv_features as py_f

__all__ = [
    'glcm',
    'glcm_props',
]


########################################################################################################################

def glcm(
        gray_image: np.ndarray, distances: np.ndarray, angle: np.ndarray, levels: int | None = None,
        symmetric: bool = False, normalize: bool = False
) -> np.ndarray:
    return py_f.gray_co_occurrence_matrix(gray_image, distances, angle, levels, symmetric, normalize)


def glcm_props(
        p: np.ndarray,
        CON: bool = True,
        ASM: bool = True,
        IDM: bool = True,
        COR: bool = True
) -> dict:
    p = np_compliance(p, arg_name='p', _check_finite=True)
    if p.ndim != 4 or p.shape[0] != p.shape[1]:
        raise TypeError('invalid glcm (p) shape expected to be (levels, levels, distances, angle)')
    p = p.astype(np.float64)
    s = np.sum(p, axis=(0, 1), keepdims=True)
    s[s == 0] = 1
    p_norm = p / s

    i = np.arange(p.shape[0], dtype=np.float64)[..., np.newaxis]
    j = i.reshape((1, -1))

    out = dict()

    if CON:
        out['CON'] = np.sum(((i - j) ** 2)[..., np.newaxis, np.newaxis] * p_norm, axis=(0, 1))

    if ASM:
        out['ASM'] = np.sum(p_norm ** 2, axis=(0, 1))

    if IDM:
        out['IDM'] = np.sum(p_norm / (1. + (i - j) ** 2)[..., np.newaxis, np.newaxis], axis=(0, 1))

    if COR:
        i = i[..., np.newaxis, np.newaxis]
        j = j[..., np.newaxis, np.newaxis]

        di = i - np.sum(p_norm * i, axis=(0, 1))
        dj = j - np.sum(p_norm * j, axis=(0, 1))

        std_i = np.sqrt(np.sum(p_norm * (di ** 2), axis=(0, 1)))
        std_j = np.sqrt(np.sum(p_norm * (dj ** 2), axis=(0, 1)))

        cor = np.sum(di * dj * p_norm, axis=(0, 1))
        non_zero = (std_i * std_j) > 0
        cor[non_zero] = cor[non_zero] / (std_i[non_zero] * std_j[non_zero])
        cor[~non_zero] = 1

        out['COR'] = cor

    return out

########################################################################################################################
