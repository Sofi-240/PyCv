import numpy as np
import abc
from typing import Any
from pycv._lib.array_api.regulator import np_compliance

__all__ = [
    "get_src_from_shape",
    "ProjectiveTransform",
    "RidgeTransform",
    "SimilarityTransform",
    "AffineTransform"
]

"""
________________________________________________________________________________________________________________________
Ridge Transformation - Translate, Rotate.

Ridge transformation includes rotation, and transformation. 
It preserves distances between points and angles, making it a rigid transformation. 
In other words, Ridge transformations maintain the shape and size of objects in the transformed space:

    | cos(theta), sin(theta), tx |
    | sin(theta), cos(theta), ty |
    |     0     ,      0    ,  1 |

where theta is the rotation angle and (tx, ty) is the translation vector. 
________________________________________________________________________________________________________________________
Similarity transformation - Scale (uniform), Translate, Rotate.

A similarity transformation, also known as a scaling and rotation transformation, 
preserves the shape of an object while allowing for uniform scaling and rotation. 
It includes translation, rotation, and scaling components. 
The transformation matrix for 2D similarity transformations looks like:

    | s*cos(theta), -s*sin(theta), tx |
    | s*sin(theta),  s*cos(theta), ty |
    |     0       ,     0       ,  1  |

where s is the scaling factor, theta is the rotation angle and (tx, ty) is the translation vector. 
________________________________________________________________________________________________________________________

Affine transformation - Scale, Shear, Translate, Rotate.

An affine transformation includes scaling, rotation, translation, and shear. 
It preserves straight lines and parallelism. 
The transformation matrix for 2D affine transformations is:

    | a, b, tx |       | a0, a1, a2 |
    | c, d, ty |       | b0, b1, b2 |
    | 0, 0, 1  |       |  0,  0,  1 |

where a, b, c, d are scaling, rotation, and shear parameters and (tx, ty) is the translation vector.

________________________________________________________________________________________________________________________
Projective transformation (homography).

A projective transformation, often referred to as a homography, is a more general transformation 
that includes perspective distortions. It maps a set of coplanar points from one 
image to another while allowing for perspective distortions. 
The transformation matrix for 2D homography is:

    | h11, h12, h13 |
    | h21, h22, h23 |
    | h31, h32, h33 |

Homographies have eight degrees of freedom, and they can represent a 
wide range of transformations, including perspective transformations.    

"""


########################################################################################################################

def _get_rotation_matrix(
        nd: int,
        angle: float | tuple,
        degree: bool = False
) -> np.ndarray:
    """
    Generate rotation matrix.

    for 2D:
        R = [[ cos(angle), -sin(angle) ]
            [ sin(angle),  cos(angle) ]]

    for 3D:
        cos[i] = cos(angle[i]);  sin[i] = sin(angle[i])

        Rx = [[   1   ,    0,      1   ],
              [   0   , cos[0], -sin[0]],
              [   0   , sin[0], cos[0]]]

        Ry = [[cos[1] ,   0   , sin[1] ],
              [   0   ,   1   ,    0   ],
              [-sin[1],   0   , cos[1] ]]

        Rz = [[cos[2], -sin[2],   0    ],
              [sin[2], cos[2] ,   0    ],
              [  0   ,   0    ,   1    ]]

        R = Rx @ Ry @ Rz

    Parameters
    ----------
    nd : int
        Number of dimensions can be 2 or 3.
    angle : float, tuple
        the rotation angle for 2d need to be scalar, and for 3d rotation angle need to be
        a tuple with size of 3 or scalar than the same angle is applied to all dimensions.
    degree : bool, optional
        If the given angle is in degree format.

    Returns
    -------
    rotation matrix : np.ndarray of floats

    """

    def _gen_matrix(c: float, s: float) -> np.ndarray:
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    if nd not in {2, 3}:
        raise ValueError('nd can be 2 or 3')

    if not ((nd == 2 and np.isscalar(angle)) or (nd == 3 and (np.isscalar(angle) or len(angle) == 3))):
        raise ValueError(
            'for 2d rotation angle need to be scalar, and for 3d rotation angle need to be a tuple with size of 3 or scalar')

    if degree:
        angle = np.deg2rad(angle)

    if nd == 2:
        return _gen_matrix(np.cos(angle), np.sin(angle))

    if np.isscalar(angle):
        angle = np.array([angle] * 3, dtype=np.float64)
    else:
        angle = np.asarray(angle, dtype=np.float64)

    cos, sin = np.cos(angle), np.sin(angle)
    sin[1] *= -1

    out = np.eye(3, dtype=np.float64)

    for i in range(3):
        put_axis = tuple(a for a in range(3) if a != i)
        slc = (
            (put_axis[0], put_axis[0], put_axis[1], put_axis[1]),
            (put_axis[0], put_axis[1], put_axis[0], put_axis[1])
        )
        euler_rot = np.eye(3, dtype=np.float64)
        euler_rot[slc] = _gen_matrix(cos[i], sin[i]).ravel()
        out = out @ euler_rot

    return out


def _get_scale_matrix(
        nd: int,
        scale: float | tuple,
        axis: int | tuple | None = None
) -> np.ndarray:
    if axis is None:
        axis = tuple(range(nd))
    if np.isscalar(axis):
        axis = (axis,)

    if not all(a < nd for a in axis):
        raise ValueError('axis is out of range for the given ndim')

    if np.isscalar(scale):
        scale = tuple(scale if a in axis else 1 for a in range(nd))
    elif len(scale) != len(axis):
        raise ValueError('scale and axis has different size')
    else:
        scale_fix = [1] * nd
        scale_iter = iter(scale)
        for i, ax in enumerate(axis):
            scale_fix[ax] = next(scale_iter)
        scale = scale_fix

    out = np.eye(nd, dtype=np.float64) * np.asarray(scale, dtype=np.float64)

    return out


def _get_translation_matrix(
        nd: int,
        translation: float | tuple,
        axis: int | tuple | None = None
) -> np.ndarray:
    if axis is None:
        axis = tuple(range(nd))
    if np.isscalar(axis):
        axis = (axis,)

    if not all(a < nd for a in axis):
        raise ValueError('axis is out of range for the given ndim')

    if np.isscalar(translation):
        scale = tuple(translation if a in axis else 1 for a in range(nd))
    elif len(translation) != len(axis):
        raise ValueError('translation and axis has different size')
    else:
        t_fix = [0] * nd
        t_iter = iter(translation)
        for i, ax in enumerate(axis):
            t_fix[ax] = next(t_iter)
        translation = t_fix

    out = np.eye(nd + 1, dtype=np.float64)
    out[:nd, nd] = np.asarray(translation, dtype=np.float64)
    return out


def get_src_from_shape(
        shape: tuple
) -> np.ndarray:
    nd = len(shape)
    src = np.reshape(np.indices(shape), (nd, -1))
    if nd == 1:
        return src
    src = np.transpose(src, (1, 0))
    src[:, [0, 1]] = src[:, [1, 0]]
    src = np.concatenate([src, np.ones((src.shape[0], 1))], axis=1)
    return src


def _build_affine_matrix(
        nd: int = 2,
        rotation: float | tuple | None = None,
        translation: float | tuple | None = None,
        scale: float | tuple | None = None,
        shear: float | tuple | None = None,
) -> np.ndarray:
    assert nd == 2

    if rotation is None:
        rotation = (0, 0)
    elif np.isscalar(rotation):
        rotation = (rotation, rotation)
    elif len(rotation) != nd:
        raise ValueError('rotation size need to be equal to ndim')

    if translation is None:
        translation = (0, 0)
    elif np.isscalar(translation):
        translation = (translation, translation)
    elif len(translation) != nd:
        raise ValueError('translation size need to be equal to ndim')

    if scale is None:
        scale = (1, 1)
    elif np.isscalar(scale):
        scale = (scale, scale)
    elif len(scale) != nd:
        raise ValueError('scale size need to be equal to ndim')

    if shear is None:
        shear = (0, 0)
    elif np.isscalar(shear):
        shear = (shear, 0)
    elif len(shear) != nd:
        raise ValueError('shear size need to be equal to ndim')

    rx, ry = np.deg2rad(rotation)
    tx, ty = translation
    sx, sy = scale
    shx, shy = shear

    a0 = sx * (np.cos(rx) + np.tan(shy) * np.sin(rx))
    a1 = -sy * (np.cos(rx) * np.tan(shx) + np.sin(rx))
    a2 = tx

    b0 = sx * (np.sin(rx) - np.tan(shy) * np.cos(rx))
    b1 = -sy * (np.sin(rx) * np.tan(shx) - np.cos(rx))
    b2 = ty

    out = np.array(
        [[a0, a1, a2],
         [b0, b1, b2],
         [0, 0, 1]], dtype=np.float64
    )

    return out


def _valid_src_for_transform(src: np.ndarray, nd: int) -> np.ndarray:
    src = np_compliance(src, 'src', _check_finite=True).astype(np.float64)

    if src.ndim != 2 or src.shape[1] not in {nd, nd + 1}:
        raise ValueError('Invalid scr shape expected to be (N, nd or nd + 1)')

    if src.shape[1] == nd:
        src = np.concatenate([src, np.ones((src.shape[0], 1))], axis=1)

    return src


########################################################################################################################

class _BaseGeometricTransform(abc.ABC):

    @abc.abstractmethod
    def __call__(self, src: np.ndarray) -> np.ndarray:
        """
        Apply forward transformation.

        Parameters
        ----------
        src : (N, ndim or ndim + 1) array_like
            Source coordinates.

        Returns
        -------
        dst : (N, ndim + 1) array
            Destination coordinates.
        """

    @abc.abstractmethod
    def __repr__(self):
        """
        Return str abstract of the class.
        """

    @abc.abstractmethod
    def __array__(self) -> np.ndarray:
        """
        Return transform matrix.
        """

    @property
    @abc.abstractmethod
    def ndim(self) -> int:
        """
        Return the dimension of the transform
        """

    @abc.abstractmethod
    def transform(self, src: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Apply forward transformation.

        Parameters
        ----------
        src : (N, ndim or ndim + 1) array_like
            Source coordinates.
        matrix : (ndim + 1, ndim + 1) array_like
            Transform matrix.

        Returns
        -------
        dst : (N, ndim + 1) array
            Destination coordinates.
        """


class ProjectiveTransform(_BaseGeometricTransform):
    def __init__(self, nd: int = 2, matrix: np.ndarray | None = None):
        if matrix is None:
            self.nd = nd
            self.matrix = np.eye(nd + 1, dtype=np.float64)
        else:
            matrix = np_compliance(matrix, 'Matrix', _check_finite=True)
            self.nd = matrix.shape[0] - 1
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError('Invalid matrix shape expected to be (nd + 1, nd + 1)')
            self.matrix = matrix

    def __call__(self, src: np.ndarray, inverse: bool = False) -> np.ndarray:
        src = _valid_src_for_transform(src, self.ndim)
        matrix = self.matrix if not inverse else self.inverse_matrix
        return self._transform(src, matrix)

    def __repr__(self):
        _out = f"{self.__class__.__name__}: nd={self.ndim}, " \
               f"matrix = \n{np.array2string(self.matrix, separator=', ')}"
        return _out

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            matrix = np_compliance(other, 'Matrix', _check_finite=True)
            if matrix.shape != self.matrix.shape:
                raise ValueError(f'Invalid matrix shape expected to be {self.matrix.shape}')
            matrix = matrix.astype(self.matrix.dtype)
        elif isinstance(other, self.__class__):
            if other.matrix.shape != self.matrix.shape:
                raise ValueError(f'Invalid matrix shape expected to be {self.matrix.shape}')
            matrix = other.matrix.astype(self.matrix.dtype)
        else:
            raise TypeError('Cannot add transform from transformer of different type')
        return type(self)(matrix=matrix @ self.matrix)

    def __iadd__(self, other):
        self.add_transform(other)
        return self

    def __eq__(self, other):
        if isinstance(other, np.ndarray):
            return other.shape == self.matrix.shape and np.all(other == self.matrix)
        if isinstance(other, self.__class__):
            return other.matrix.shape == self.matrix.shape and np.all(other.matrix == self.matrix)
        return False

    def __array__(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float64)

    @property
    def ndim(self) -> int:
        return self.nd

    @property
    def inverse_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)

    def _transform(self, src: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        nd = matrix.shape[0] - 1

        dst = src @ matrix.T
        dst[dst[:, nd] == 0, nd] = np.finfo(float).eps
        dst[:, :nd] /= dst[:, nd:nd + 1]

        return dst

    def transform(self, src: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        matrix = np_compliance(matrix, 'matrix', _check_finite=True).astype(np.float64)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Invalid matrix shape expected to be (nd + 1, nd + 1)')

        src = _valid_src_for_transform(src, matrix.shape[0] - 1)
        return self._transform(src, matrix)

    def add_transform(self, other: Any) -> None:
        if isinstance(other, np.ndarray):
            matrix = np_compliance(other, 'Matrix', _check_finite=True)
            if matrix.shape != self.matrix.shape:
                raise ValueError(f'Invalid matrix shape expected to be {self.matrix.shape}')
            matrix = matrix.astype(self.matrix.dtype)
        elif isinstance(other, self.__class__):
            if other.matrix.shape != self.matrix.shape:
                raise ValueError(f'Invalid matrix shape expected to be {self.matrix.shape}')
            matrix = other.matrix.astype(self.matrix.dtype)
        else:
            raise TypeError('Cannot add transform from transformer of different type')
        self.matrix = matrix @ self.matrix


class RidgeTransform(ProjectiveTransform):
    def __init__(
            self,
            nd: int = 2,
            rotation: float | tuple | None = None,
            translation: float | tuple | None = None,
            matrix: np.ndarray | None = None
    ):
        super().__init__(nd, matrix=matrix)
        if matrix is None:
            if rotation is not None:
                self.matrix[:nd, :nd] = _get_rotation_matrix(self.nd, rotation, True)

            if translation is not None:
                self.matrix[:nd, nd] = _get_translation_matrix(self.nd, translation)[:nd, nd]


class SimilarityTransform(RidgeTransform):
    def __init__(
            self,
            nd: int = 2,
            rotation: float | tuple | None = None,
            translation: float | tuple | None = None,
            scale: float | None = None,
            matrix: np.ndarray | None = None
    ):
        super().__init__(nd, rotation=rotation, translation=translation, matrix=matrix)
        if matrix is None:
            if scale is not None:
                self.matrix[:nd, :nd] *= scale


class AffineTransform(ProjectiveTransform):
    def __init__(
            self,
            nd: int = 2,
            rotation: float | tuple | None = None,
            translation: float | tuple | None = None,
            scale: float | tuple | None = None,
            shear: float | tuple | None = None,
            matrix: np.ndarray | None = None
    ):
        super().__init__(nd, matrix=matrix)
        assert self.nd == 2
        if matrix is None:
            self.matrix = _build_affine_matrix(
                self.nd, rotation=rotation, translation=translation, scale=scale, shear=shear
            )

########################################################################################################################
