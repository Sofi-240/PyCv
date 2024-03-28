import numpy as np
from .._lib.array_api.regulator import np_compliance
from pycv._lib._src.c_pycv import CMinMaxTree

__all__ = [
    "MaxTree",
    "MinTree"
]


########################################################################################################################

class _MinMaxTree(CMinMaxTree):
    def __init__(self, data: np.ndarray, connectivity: int = 1, max_tree: bool = True):
        data = np_compliance(data, 'data', _check_finite=True, _check_atleast_nd=1)
        super().__init__(data, connectivity=connectivity, max_tree=int(max_tree))

    def tree_filter(self, values_map, threshold):
        if not isinstance(values_map, np.ndarray):
            raise RuntimeError('currently supported values map are numpy.ndarray')
        values_map = np_compliance(values_map, 'values_map', _check_finite=True)
        if values_map.shape != self.dims:
            raise ValueError('values map shape need to be equal to data shape')
        return super().tree_filter(values_map, threshold)


########################################################################################################################

class MaxTree(_MinMaxTree):
    def __init__(self, data: np.ndarray, connectivity: int = 1):
        super().__init__(data, connectivity, True)


class MinTree(_MinMaxTree):
    def __init__(self, data: np.ndarray, connectivity: int = 1):
        super().__init__(data, connectivity, False)

########################################################################################################################
