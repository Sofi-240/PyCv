import numpy as np
import warnings
from typing import Iterable
from types import DynamicClassAttribute
from ._measure import perimeter, moments
from ..morphological import ConvexHull, binary_fill_holes, binary_edge
from .._lib.array_api.regulator import np_compliance
from .._lib.array_api.dtypes import as_binary_array

__all__ = [
    'Bbox', 'RegionProperties'
]

########################################################################################################################

_NotAllowedNames = {
    '_use_cache_', '_cache_', '_properties_names_',
}


########################################################################################################################

class region_property(DynamicClassAttribute):
    def __set_name__(self, owner, name):
        self._name = name
        self._clsname = owner.__name__

    def __get__(self, instance, owner):
        if instance is None or self.fget is None:
            raise AttributeError(f'{self._clsname} has no attribute {self._name}')
        _use_cache_ = getattr(instance, '_use_cache_', False)
        if not _use_cache_:
            return self.fget(instance)
        if self._name in instance._cache_:
            return instance._cache_[self._name]
        instance._cache_[self._name] = self.fget(instance)
        return instance._cache_[self._name]

    def __set__(self, instance, value):
        if self.fset is None:
            raise AttributeError(f'{self._clsname} cannot set attribute to {self._name}')
        _use_cache_ = getattr(instance, '_use_cache_', False)
        if not _use_cache_ or self._name not in instance._cache_:
            return self.fset(instance, value)
        instance._cache_.pop(self._name)
        self.fset(instance, value)
        instance._cache_[self._name] = self.fget(instance)

    def __delete__(self, instance):
        if self.fdel is None:
            raise AttributeError(f'{self._clsname} cannot delete {self._name} attribute')
        _use_cache_ = getattr(instance, '_use_cache_', False)
        if not _use_cache_ or self._name not in instance._cache_:
            return self.fdel(instance)
        instance._cache_.pop(self._name)
        return self.fdel(instance)


########################################################################################################################

class properties_dict(dict):
    def __init__(self, clsname: str):
        super().__init__()
        self.clsname = clsname
        self.properties = {}

    def __setitem__(self, key, value):
        if isinstance(value, (region_property, property, DynamicClassAttribute)):
            self.properties[key] = None
        super().__setitem__(key, value)


class properties(type):
    @classmethod
    def __prepare__(mcs, cls, bases):
        clsdict = properties_dict(cls)
        return clsdict

    def __new__(mcs, cls, bases, clsdict):
        _properties_names_ = set(p for p in clsdict.properties)
        invalid_attrs = _properties_names_ & _NotAllowedNames
        if invalid_attrs:
            raise ValueError(f'got invalid attributes in {cls}: {", ".join(invalid_attrs)}\n '
                             f'dont use any of {", ".join(_NotAllowedNames)}')

        clsdict['_properties_names_'] = _properties_names_
        return super().__new__(mcs, cls, bases, clsdict)

    def __repr__(cls):
        return f'{cls.__name__}'

    def __contains__(cls, name):
        return name in cls._properties_names_

    def __bool__(self):
        return True

    def __getitem__(cls, name):
        _properties_names_ = cls.__dict__.get('_properties_names_', {})
        try:
            return getattr(cls, name)
        except KeyError:
            raise AttributeError(f'{name} is not a member of {cls}') from None


########################################################################################################################


class Properties(metaclass=properties):

    def __init__(self, *args, **kwargs):
        self._use_cache_ = False
        self._cache_ = dict()

    def __getitem__(self, name):
        if name in self._properties_names_:
            try:
                return getattr(self, name)
            except NotImplementedError as _exp:
                warnings.warn(f"{_exp} returning None")
                return None
        raise AttributeError(f'{name} is not a member of {self.__class__.__name__}') from None

    def __contains__(self, name):
        return name in self._properties_names_

    def __iter__(self):
        return ((p, self[p]) for p in self._properties_names_)


########################################################################################################################


class Bbox(Properties):

    def __init__(self, bbox: tuple):
        super().__init__()
        if all(isinstance(b, slice) for b in bbox):
            self.bbox = [
                np.array(tuple(b.start for b in bbox), dtype=np.int64),
                np.array(tuple(b.stop - 1 for b in bbox), dtype=np.int64)
            ]
        else:
            if len(bbox) != 2:
                raise ValueError('bbox must be tuple with size of 2 (top left, bottom right)')
            self.bbox = list(np.array(b, dtype=np.int64) for b in bbox)

    def __repr__(self):
        return f'{self.__class__.__name__}: ' \
               f'{np.array2string(self.top_left, separator=", ")}, {np.array2string(self.bottom_right, separator=", ")}'

    def __call__(self, image: np.ndarray):
        if image.ndim not in (self.ndim, self.ndim + 1):
            raise ValueError(f'invalid image dimensions expected to be '
                             f'{self.ndim} or {self.ndim + 1} for color image')
        if any(b >= s for b, s in zip(self.bottom_right, image.shape)):
            raise ValueError(f'image dimensions length is to small')
        return image[self.slice]

    @property
    def ndim(self):
        return self.top_left.shape[0]

    @property
    def top_left(self) -> np.ndarray:
        return self.bbox[0]

    @property
    def bottom_right(self) -> np.ndarray:
        return self.bbox[1]

    @region_property
    def centroid(self) -> np.ndarray:
        return sum(self.bbox) / 2

    @region_property
    def centroid_local(self) -> np.ndarray:
        return self.centroid - self.top_left

    @region_property
    def centroid_point(self) -> np.ndarray:
        return (self.centroid + 0.5).astype(np.int64)

    @region_property
    def centroid_point_local(self) -> np.ndarray:
        return (self.centroid_local + 0.5).astype(np.int64)

    @region_property
    def area(self) -> int:
        return int(np.prod(self.bottom_right - self.top_left + 1))

    @region_property
    def slice(self) -> tuple[slice]:
        return tuple(slice(s, e + 1) for s, e in zip(*self.bbox))


########################################################################################################################


class RegionProperties(Properties):
    def __init__(
            self,
            label_image: np.ndarray,
            intensity_image: np.ndarray | None = None,
            offset: Iterable | Bbox | None = None,
            label: int | None = None,
            use_cache: bool = False
    ):
        super().__init__()
        label_image = np_compliance(label_image, 'label_image')
        self._label_image_ = as_binary_array(label_image)
        self._label_ = label if label is not None else np.max(label_image)
        if intensity_image is not None:
            intensity_image = np_compliance(label_image, 'label_image', _check_finite=True)
            if intensity_image.ndim == self.ndim:
                intensity_image = np.expand_dims(intensity_image, -1)
            if intensity_image.ndim != self.ndim + 1 or intensity_image.shape[:-1] != self.shape:
                raise ValueError(f'invalid intensity image shape '
                                 f'expected to be ({", ".join(self.shape)}, channels)')
        self._intensity_image_ = intensity_image
        if isinstance(offset, Bbox):
            if offset.ndim != self.ndim:
                raise ValueError('offset size need to be equal to label image ndim')
            if tuple(br - tp + 1 for br, tp in zip(offset.bottom_right, offset.top_left)) != self.shape:
                raise ValueError('bbox size is not equal to image shape')
            self._bbox_ = offset
        else:
            if offset is not None:
                offset = tuple(i for i in offset)
            else:
                offset = (0,) * self.ndim

            if len(offset) != self.ndim:
                raise ValueError('offset size need to be equal to label image ndim')

            self._bbox_ = Bbox((offset, tuple(o + s for o, s in zip(offset, self.shape))))
        self._convex_hull_ = None
        self._use_cache_ = use_cache

    @property
    def shape(self) -> tuple:
        return self._label_image_.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def label(self):
        return self._label_

    @property
    def offset(self) -> np.ndarray:
        return self.bbox.top_left

    @offset.setter
    def offset(self, _offset: Iterable):
        _offset = tuple(i for i in _offset)
        if len(_offset) != self.ndim:
            raise ValueError('offset size need to be equal to label image ndim')
        self._bbox_ = Bbox((_offset, tuple(o + s for o, s in zip(_offset, self.shape))))

    @property
    def slice(self) -> tuple:
        return self.bbox.slice

    @property
    def label_image(self):
        return self._label_image_

    @property
    def bbox(self):
        return self._bbox_

    @property
    def intensity_image(self):
        if self._intensity_image_ is None:
            raise NotImplementedError(f"intensity image wasn't specified.")
        return self._intensity_image_

    @region_property
    def area(self) -> int:
        return np.sum(self._label_image_)

    @region_property
    def bbox_area(self) -> int:
        return self.bbox.area

    @region_property
    def pixels_list_local(self) -> np.ndarray:
        return np.stack(self._label_image_.nonzero(), axis=-1)

    @region_property
    def pixels_list(self) -> np.ndarray:
        return self.pixels_list_local + self.offset

    @region_property
    def centroid_local(self) -> np.ndarray:
        return self.pixels_list_local.mean(axis=0)

    @region_property
    def centroid(self) -> np.ndarray:
        return self.centroid_local + self.offset

    @region_property
    def convex_hull(self) -> ConvexHull:
        if self.ndim != 2:
            raise NotImplementedError(f'property convex_image supported just for 2D images')
        if self._convex_hull_ is None:
            edge_points = np.stack(binary_edge(self._label_image_).nonzero(), axis=-1)
            self._convex_hull_ = ConvexHull(points=edge_points, image_shape=self.shape)
        return self._convex_hull_

    @region_property
    def convex_image(self) -> np.ndarray:
        return self.convex_hull.to_image()

    @region_property
    def area_convex(self) -> int:
        return np.sum(self.convex_image)

    @region_property
    def filled_image(self) -> np.ndarray:
        return binary_fill_holes(self._label_image_)

    @region_property
    def area_filled(self) -> int:
        return np.sum(self.filled_image)

    @region_property
    def perimeter(self) -> float:
        return perimeter(self._label_image_)

    @region_property
    def intensity_max(self) -> np.ndarray:
        return np.amax(self.intensity_image, axis=-1)

    @region_property
    def intensity_min(self) -> np.ndarray:
        return np.amin(self.intensity_image, axis=-1)

    @region_property
    def euler_number(self) -> float:
        return self.area - self.area_filled

    @region_property
    def circularity(self) -> float:
        return (self.perimeter ** 2) / (np.pi * 4 * self.area_filled)

    @region_property
    def moments(self) -> np.ndarray:
        return moments(self._label_image_, order=3)

    @region_property
    def moments_central(self) -> np.ndarray:
        return moments(self._label_image_, center=self.centroid_local, order=3)

    @region_property
    def intensity_moments(self) -> np.ndarray:
        _intensity_image_ = self.intensity_image
        return np.stack([moments(_intensity_image_[..., i], order=3) for i in range(_intensity_image_.ndim)], axis=-1)

    @region_property
    def intensity_moments_central(self) -> np.ndarray:
        _intensity_image_ = self.intensity_image
        return np.stack([moments(_intensity_image_[..., i], center=self.centroid_local, order=3) for i in range(_intensity_image_.ndim)], axis=-1)


########################################################################################################################
