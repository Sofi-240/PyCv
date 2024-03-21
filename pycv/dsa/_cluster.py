import numpy as np
from .._lib.array_api.regulator import np_compliance
from pycv._lib._src.c_pycv import CKMeans

__all__ = [
    "KMeans"
]


########################################################################################################################

class KMeans(CKMeans):
    def __init__(
            self,
            data: np.ndarray | None = None,
            k: int = 4,
            iterations: int = 300,
            tol: float = 0.00001,
            init_method: str = 'kmeans++',
            pnorm: int = 2
    ):
        if init_method not in ('random', 'kmeans++'):
            raise ValueError('unsupported init method use `kmeans++` or `random`')
        init_method = 2 if init_method == 'kmeans++' else 1
        super().__init__(k=k, iterations=iterations, tol=tol, init_method=init_method)
        if data is not None:
            self.fit(data, pnorm)

    def __repr__(self):
        _init = ('random', 'kmeans++')
        return f'{self.__class__.__name__}: ' \
               f'K={self.k}, init={_init[self.init_method - 1]}, n_features={self.ndim}'

    def fit(self, data: np.ndarray, pnorm: int = 2):
        data = np_compliance(data, 'data', _check_finite=True)
        if data.ndim != 2:
            raise ValueError('data must be of shape (n [samples], m [features])')
        data = data.astype(np.float64)
        if data.shape[0] < self.k:
            raise RuntimeError("k is higher then the given n samples in the data")
        super().fit(data, pnorm=float(pnorm))

    def predict(self, data: np.ndarray, pnorm: int = 2):
        if self.data is None:
            raise RuntimeError("KMeans was not fitted")
        data = np_compliance(data, 'data', _check_finite=True)
        if data.ndim != 2 or data.shape[1] != self.ndim:
            raise ValueError('data must be of shape (n [samples], m [features equal to ndim])')
        data = data.astype(np.float64)
        return super().predict(data, pnorm=float(pnorm))


########################################################################################################################
