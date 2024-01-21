from pycv._lib.filters_support.thresholding import otsu, kapur, li_and_lee, minimum, minimum_error, mean, adaptive
from types import FunctionType

__all__ = [
    'get_method_function',
]


########################################################################################################################

def get_method_function(method: str) -> FunctionType:
    supported_mode = {
        'otsu': otsu,
        'kapur': kapur,
        'li_and_lee': li_and_lee,
        'minimum': minimum,
        'minimum_error': minimum_error,
        'mean': mean,
        'adaptive': adaptive
    }
    out = supported_mode.get(method, None)
    if out is None:
        raise ValueError(f'{method} is not supported use {list(supported_mode.keys())}')
    return out

########################################################################################################################
