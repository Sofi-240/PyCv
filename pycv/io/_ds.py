import numpy as np
from pycv.io._rw import load_image

__all__ = [
    'ImageCollection'
]


########################################################################################################################

class ImageCollection(object):
    def __init__(self, path: str | list[str], load_func=None, use_cache: bool = False):
        self.files = [path] if isinstance(path, str) else path
        self._loads = load_func if load_func is not None else load_image
        self.use_cache = use_cache
        if use_cache:
            self.collection = np.empty((len(self.files),), dtype=object)
        else:
            self.collection = np.empty((1,), dtype=object)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.collection}'

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(f'index {index} is out of range for size of {len(self)}')
        if self.use_cache and self.collection[index] is not None:
            return self.collection[index]
        out = self._loads(self.files[index])
        if self.use_cache:
            self.collection[index] = out
        else:
            self.collection[0] = out
        return out

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i == len(self):
            raise StopIteration
        self._i += 1
        return self[self._i - 1]

    def concatenate(self):
        try:
            out = np.stack(tuple(arr for arr in self), axis=0)
        except ValueError:
            raise ValueError('images has different shapes')
        return out

########################################################################################################################
