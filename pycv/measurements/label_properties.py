from .._lib._properties_struct import Properties, region_property
from typing import Iterable
import numpy as np
from .measure import perimeter, moments
from ..morphological import ConvexHull, binary_fill_holes, binary_edge, find_object
from .._lib.array_api.dtypes import as_binary_array

__all__ = [
    'Bbox',
    'NBbox',
    'RegionProperties',
    'NRegionProperties'
]


########################################################################################################################

class _BaseBbox(Properties):
    def __new__(cls, bbox=None):
        obj = super().__new__(cls, _use_cache=False)
        if obj._sequence_:
            return obj

        if not bbox:
            raise TypeError(
                f'{cls.__name__} new missing 1 required positional argument: `bbox`'
            )
        if all(isinstance(b, slice) for b in bbox):
            bbox = [
                np.array(tuple(b.start for b in bbox), dtype=np.int64),
                np.array(tuple(b.stop - 1 for b in bbox), dtype=np.int64)
            ]
        else:
            if len(bbox) != 2:
                raise ValueError('bbox must be tuple with size of 2 (top left, bottom right)')
            bbox = list(np.array(b, dtype=np.int64) for b in bbox)
        setattr(obj, 'bbox', bbox)
        return obj

    def __repr__(self):
        if self._sequence_:
            return super().__repr__()
        return f'{self.__class__.__name__}: ' \
               f'{np.array2string(self.top_left, separator=", ")}, {np.array2string(self.bottom_right, separator=", ")}'

    def __call__(self, image: np.ndarray):
        if self._sequence_:
            return [b(image) for b in self._items_]
        if image.ndim not in (self.ndim, self.ndim + 1):
            raise ValueError(f'invalid image dimensions expected to be '
                             f'{self.ndim} or {self.ndim + 1} for color image')
        if any(b >= s for b, s in zip(self.bottom_right, image.shape)):
            raise ValueError(f'image dimensions length is to small')
        return image[self.slice]

    @region_property(_cache=True)
    def top_left(self):
        return self.bbox[0]

    @region_property(_cache=True)
    def bottom_right(self):
        return self.bbox[1]

    @region_property(_cache=False)
    def ndim(self):
        return self.top_left.shape[0]

    @region_property(_cache=True)
    def centroid(self) -> np.ndarray:
        return sum(self.bbox) / 2

    @region_property(_cache=True)
    def centroid_local(self) -> np.ndarray:
        return self.centroid - self.top_left

    @region_property(_cache=True)
    def centroid_point(self) -> np.ndarray:
        return (self.centroid + 0.5).astype(np.int64)

    @region_property(_cache=True)
    def centroid_point_local(self) -> np.ndarray:
        return (self.centroid_local + 0.5).astype(np.int64)

    @region_property(_cache=True)
    def area(self) -> int:
        return int(np.prod(self.bottom_right - self.top_left + 1))

    @region_property(_cache=False)
    def slice(self) -> tuple[slice]:
        return tuple(slice(s, e + 1) for s, e in zip(*self.bbox))


########################################################################################################################

class Bbox(_BaseBbox):
    def __init__(self, bbox: tuple):
        pass


class NBbox(_BaseBbox, sequence=True):
    def __init__(self):
        pass


########################################################################################################################

class _BaseRegionProperties(Properties):

    def __new__(cls, label_image=None, intensity_image=None, offset=None, label=None, use_cache=False):
        obj = super().__new__(cls, _use_cache=use_cache)
        if obj._sequence_:
            obj._use_cache_ = False
            return obj

        _label_image_ = np.asarray(label_image)
        if _label_image_.ndim < 2:
            raise TypeError(f'Input {label_image} need to be with atleast 2 dimensions got {_label_image_.ndim}')
        try:
            _label_image_ = as_binary_array(_label_image_)
        except ValueError as err:
            if label is None or np.sum(_label_image_ == label) == 0:
                _label_image_ = _label_image_.astype(bool)
            else:
                _label_image_ = _label_image_ == label
        _label_ = label if label is not None else np.max(_label_image_)

        setattr(obj, '_label_image_', _label_image_)
        setattr(obj, '_label_', _label_)

        if isinstance(offset, Bbox):
            if offset.ndim != obj.ndim:
                raise ValueError('offset size need to be equal to label image ndim')
            if tuple(br - tp + 1 for br, tp in zip(offset.bottom_right, offset.top_left)) != obj.shape:
                raise ValueError('bbox size is not equal to image shape')
            _bbox_ = offset
        else:
            if offset is not None:
                offset = tuple(i for i in offset)
            else:
                offset = (0,) * obj.ndim

            if len(offset) != obj.ndim:
                raise ValueError('offset size need to be equal to label image ndim')

            _bbox_ = Bbox((offset, tuple(o + s for o, s in zip(offset, obj.shape))))

        setattr(obj, '_bbox_', _bbox_)

        if intensity_image is not None:
            _intensity_image_ = np.asarray_chkfinite(intensity_image)

            if _intensity_image_.ndim == obj.ndim:
                _intensity_image_ = np.expand_dims(_intensity_image_, -1)

            if _intensity_image_.ndim != obj.ndim + 1 or _intensity_image_.shape[:-1] != obj.shape:
                raise ValueError(f'invalid intensity image shape '
                                 f'expected to be ({", ".join(obj.shape)}, channels)')
            _multi_channels_ = _intensity_image_.shape[-1] != 1
        else:
            _intensity_image_ = None
            _multi_channels_ = False

        _convex_hull_ = None

        setattr(obj, '_intensity_image_', _intensity_image_)
        setattr(obj, '_multi_channels_', _multi_channels_)
        setattr(obj, '_convex_hull_', _convex_hull_)

        return obj

    @region_property(_cache=False)
    def shape(self) -> tuple:
        return self._label_image_.shape

    @region_property(_cache=False)
    def ndim(self) -> int:
        return len(self.shape)

    @region_property(_cache=False)
    def label(self):
        return self._label_

    @region_property(_cache=False)
    def bbox(self):
        return self._bbox_

    @region_property(_cache=False)
    def label_image(self):
        return self._label_image_

    @region_property(_cache=False)
    def intensity_image(self):
        if self._intensity_image_ is None:
            raise NotImplementedError(f"intensity image wasn't specified.")
        return self._intensity_image_

    @region_property(_cache=False)
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

    @region_property(_cache=False)
    def offset(self) -> np.ndarray:
        return self.bbox.top_left

    @region_property(_cache=False)
    def slice(self) -> tuple:
        return self.bbox.slice

    @region_property
    def filled_image(self) -> np.ndarray:
        return binary_fill_holes(self._label_image_)

    @region_property
    def area(self) -> int:
        return np.sum(self.label_image)

    @region_property
    def bbox_area(self) -> int:
        return self.bbox.area

    @region_property
    def area_convex(self) -> int:
        return np.sum(self.convex_image)

    @region_property
    def area_filled(self) -> int:
        return np.sum(self.filled_image)

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
    def perimeter(self) -> float:
        return perimeter(self._label_image_)

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
    def intensity_max(self) -> np.ndarray:
        return np.amax(self.intensity_image, axis=tuple(range(self.ndim)))

    @region_property
    def intensity_min(self) -> np.ndarray:
        return np.amin(self.intensity_image, axis=tuple(range(self.ndim)))

    @region_property
    def intensity_moments(self) -> np.ndarray:
        _intensity_image_ = self.intensity_image
        return np.stack([moments(_intensity_image_[..., i], order=3) for i in range(_intensity_image_.shape[-1])],
                        axis=-1)

    @region_property
    def intensity_moments_central(self) -> np.ndarray:
        _intensity_image_ = self.intensity_image
        return np.stack([moments(_intensity_image_[..., i], center=self.centroid_local, order=3) for i in
                         range(_intensity_image_.shape[-1])], axis=-1)


########################################################################################################################

class RegionProperties(_BaseRegionProperties):
    def __init__(
            self,
            label_image: np.ndarray,
            intensity_image: np.ndarray | None = None,
            offset: Iterable | Bbox | None = None,
            label: int | None = None,
            use_cache: bool = False
    ):
        pass


class NRegionProperties(_BaseRegionProperties, sequence=True):
    def __init__(self):
        pass

    def __call__(self, label_image: np.ndarray, intensity_image: np.ndarray | None = None, use_cache=False):
        self._items_.clear()
        for bbox, lbl in zip(find_object(label_image, as_slice=True), np.unique(label_image[label_image > 0])):
            box = Bbox(bbox)
            self.push(
                RegionProperties(
                    box(label_image),
                    box(intensity_image) if intensity_image is not None else None,
                    offset=box,
                    label=int(lbl),
                    use_cache=use_cache
                )
            )

########################################################################################################################

