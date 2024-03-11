import numpy as np
from pycv._lib.filters_support.kernel_utils import default_binary_strel
from pycv import morphological as morph

__all__ = [
    'RegionProperties',
    'region_properties'
]

########################################################################################################################

PROPS = [
    'area_object'
    'bbox',
    'area_bbox',
    'centroid',
    'local_centroid',
    'filled_image',
    'convex_hull',
    'local_convex_hull',
    'convex_image',
    'area_convex'
    'local_pixel_list',
    'pixel_list',
    'solidity',
    'extent',
]


########################################################################################################################

class RegionProperties:
    def __init__(
            self,
            label: int,
            labels_image: np.ndarray,
            intensity_image: np.ndarray | None = None,
            slc: tuple[slice] | None = None,
            connectivity: int = 1
    ):
        self.label = label
        self._image = labels_image
        self._intensity = intensity_image
        self._color_intensity = False
        if intensity_image is not None:
            ndim = labels_image.ndim
            if not (
                    intensity_image.shape[:-ndim] == labels_image.shape
                    and intensity_image.ndim in [ndim, ndim + 1]
            ):
                raise ValueError('label and intensity image has different shapes')
            if intensity_image.ndim == ndim + 1:
                self._color_intensity = True
        if slc is None:
            slc = tuple(slice(0, s) for s in self._image.shape)
        else:
            if len(slc) != labels_image.ndim:
                raise ValueError('slice need to have len equal to image ndim')
            for i, _slc in enumerate(slc):
                if not isinstance(_slc, slice):
                    raise ValueError('slice need to be tuple of slice object')
                if _slc.stop is None:
                    _slc.stop = _slc.start + labels_image.shape[i]
        self.slice = slc
        self._bbox_slice = tuple(slice(np.amin(cc), np.amax(cc) + 1) for cc in np.where(labels_image == label))
        self.offset = np.array([s.start for s in self._bbox_slice], np.int64)
        self._local_offset = np.array([s.start for s in self.slice], np.int64)
        self._connectivity = connectivity

        self._convex_hull = None

    def __repr__(self):
        return f"{self.__class__.__name__}: label:{self.label}"

    @property
    def shape(self) -> tuple:
        return self._image.shape

    @property
    def ndim(self) -> int:
        return self._image.ndim

    @property
    def object_image(self) -> np.ndarray:
        return self._image == self.label

    @property
    def area_object(self) -> int:
        return np.sum(self.object_image)

    @property
    def intensity_image(self) -> np.ndarray:
        if self._intensity is None:
            raise AttributeError('`intensity image` has not been specified')
        return self._intensity

    @property
    def bbox(self) -> tuple[list, list]:
        return (
            [si.start + sb.start for (si, sb) in zip(self.slice, self._bbox_slice)],
            [si.start + sb.stop - 1 for (si, sb) in zip(self.slice, self._bbox_slice)]
        )

    @property
    def area_bbox(self) -> int:
        return np.prod(tuple(br - tp + 1 for (tp, br) in zip(*self.bbox)))

    @property
    def local_centroid(self) -> np.ndarray:
        return np.stack(np.where(self.object_image), axis=1).mean(axis=0)

    @property
    def centroid(self) -> np.ndarray:
        return self.local_centroid + self._local_offset.astype(np.float)

    @property
    def filled_image(self) -> np.ndarray:
        se = default_binary_strel(self.ndim, self._connectivity)
        return morph.binary_fill_holes(self.object_image, strel=se)

    @property
    def area_filled(self) -> int:
        return np.sum(self.filled_image)

    @property
    def local_convex_hull(self) -> np.ndarray:
        if self._convex_hull is not None:
            return self._convex_hull
        self._convex_hull = morph.convex_hull(self.object_image, convex_img=False)
        return self._convex_hull

    @property
    def convex_hull(self) -> np.ndarray:
        if self.local_convex_hull.shape[0] > 0:
            return self.local_convex_hull + self._local_offset
        return self.local_convex_hull

    @property
    def convex_image(self) -> np.ndarray:
        if self.local_convex_hull.shape[0] == 0:
            return np.zeros_like(self.object_image)
        return morph.convex_image(self.local_convex_hull, self.shape)

    @property
    def area_convex(self) -> int:
        return np.sum(self.convex_image)

    @property
    def local_pixel_list(self) -> np.ndarray:
        return np.stack(np.where(self.object_image), axis=-1)

    @property
    def pixel_list(self) -> np.ndarray:
        if self.area_object == 0:
            return self.local_pixel_list
        return self.local_pixel_list + self._local_offset

    @property
    def solidity(self) -> float:
        return self.area_object / self.area_convex

    @property
    def extent(self) -> float:
        return self.area_object / self.area_bbox


########################################################################################################################

def region_properties(
        label_image: np.ndarray,
        mask: np.ndarray | None = None,
        intensity_image: np.ndarray | None = None
) -> list[RegionProperties]:
    objects = morph.find_object(label_image, mask, as_slice=True)

    out = []
    for slc, lbl in zip(objects, np.unique(label_image[label_image > 0])):
        out.append(
            RegionProperties(lbl, label_image[slc], intensity_image[slc] if intensity_image is not None else None, slc)
        )

    return out

########################################################################################################################
