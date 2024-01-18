import numpy as np
import typing

__all__ = [
    'np_compliance',
    'check_finite',
    'SUPPORTED_NP',
]

SUPPORTED_NP = (np.ndarray, np.generic)


########################################################################################################################

def np_compliance(
        inputs: np.ndarray,
        arg_name: str = 'array',
        _check_finite: bool = False,
) -> np.ndarray:
    msg_bad_np = arg_name + " of type s% are not supported."
    msg_bad_dtype = arg_name + " has dtype %s only boolean and numerical dtypes are supported."

    if isinstance(inputs, np.ma.MaskedArray):
        raise TypeError(msg_bad_np % 'numpy.ma.MaskedArray')
    elif isinstance(inputs, np.matrix):
        raise TypeError(msg_bad_np % 'numpy.matrix')
    elif isinstance(inputs, SUPPORTED_NP):
        dtype = inputs.dtype
        if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
            raise TypeError(msg_bad_dtype % str(dtype))
    else:
        try:
            inputs = np.asanyarray(inputs)
        except TypeError:
            raise TypeError("Inputs array not coercible by numpy")
        dtype = inputs.dtype
        if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
            raise TypeError(msg_bad_dtype % str(dtype))
    if _check_finite:
        check_finite(inputs, True, arg_name)
    return inputs


def check_finite(
        inputs: np.ndarray,
        raise_err: bool = False,
        arg_name: str = 'array'
) -> bool:
    """
    Raise exceptions on non-finite array.

    Parameters
    ----------
    inputs : numpy.ndarray
    raise_err: bool
    arg_name: str

    Returns
    -------
    None.

    Raises
    ------
    ValueError:
        If the array contain infs or NaNs.
    """
    if not isinstance(inputs, np.ndarray):
        inputs = np.asarray(inputs)
    if not np.all(np.isfinite(inputs)):
        if raise_err:
            raise RuntimeError(f'{arg_name} must not contain infs or NaNs')
        return False
    return True


########################################################################################################################
