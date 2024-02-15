import numpy as np
from pycv.measurements import find_object, convex_hull

__all__ = [
    "RegionProperties",
    "region_properties"
]


########################################################################################################################

class RegionProperties:
    def __init__(
            self,
            slc: tuple[slice],
            label: int,
            label_image: np.ndarray,
            intensity_image: np.ndarray | None = None,
            offset: tuple | None = None
    ):
        self.slice = slc
        self.label = label

        self._image = label_image == label
        if intensity_image is None:
            intensity_image = self._image
        self._intensity = intensity_image
        if offset is None:
            offset = np.zeros((self._image.ndim,), int)
        self._offsets = np.array(offset)

        self._convex_image = None
        self._convex_hull = None

    def __repr__(self):
        return f"{self.__class__.__name__}: label={self.label}"

    def __array__(self) -> np.ndarray:
        return self._image.copy()

    @property
    def shape(self) -> tuple:
        return self._image.shape

    @property
    def area(self) -> int:
        return np.sum(self._image)

    @property
    def pixel_inx_list(self):
        return np.where(self._image.ravel())[0]

    @property
    def pixel_list(self):
        return np.stack(self._image.nonzero(), axis=1)

    @property
    def bbox(self) -> list[tuple]:
        return [tuple(s.start for s in self.slice), tuple(s.stop for s in self.slice)]

    @property
    def bbox_area(self) -> int:
        return np.prod(self.shape)

    @property
    def convex_hull(self):
        if self._convex_hull is None and self.area > 3:
            self._convex_hull, self._convex_image = convex_hull(self._image, convex_image=True)
        return self._convex_hull

    @property
    def convex_image(self):
        if self._convex_hull is None and self.area > 3:
            self._convex_hull, self._convex_image = convex_hull(self._image, convex_image=True)
        return self._convex_image

    @property
    def convex_area(self) -> int:
        c = self.convex_image
        if c is None:
            return 0
        return np.sum(c)

########################################################################################################################


def region_properties(
        label_image: np.ndarray,
        mask: np.ndarray | None = None,
        intensity_image: np.ndarray | None = None
) -> list[RegionProperties]:
    objects = find_object(label_image, mask, as_slice=True)
    ll = label_image.copy()

    if mask is not None:
        ll[~mask] = 0

    out = []
    for slc, lbl in zip(objects, np.unique(ll[ll > 0])):
        out.append(
            RegionProperties(slc, lbl, ll[slc], intensity_image[slc] if intensity_image is not None else None)
        )

    return out

########################################################################################################################
