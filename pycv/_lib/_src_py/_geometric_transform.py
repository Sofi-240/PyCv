import numpy as np
import abc
from pycv._lib.array_api.regulator import np_compliance

__all__ = [
    'src_coordinates',
    'valid_src',
    'rotation_matrix',
    'translation_matrix',
    'affine_matrix',
    'ProjectiveTransform',
    'RidgeTransform',
    'SimilarityTransform',
    'AffineTransform'
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

def _valid_matrix(
        ndim: int,
        matrix: np.ndarray | None
) -> np.ndarray:
    if matrix is None:
        matrix = np.eye(ndim + 1, dtype=np.float64)
    else:
        matrix = np_compliance(matrix, 'matrix', _check_finite=True).astype(np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Invalid matrix shape expected to be (ndim + 1, ndim + 1)')
    return matrix


########################################################################################################################

def src_coordinates(
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


def valid_src(
        src: np.ndarray,
        ndim: int
) -> np.ndarray:
    src = np_compliance(src, 'src', _check_finite=True).astype(np.float64)

    if src.ndim != 2 or src.shape[1] not in {ndim, ndim + 1}:
        raise ValueError('Invalid scr shape expected to be (N, nd or nd + 1)')

    if src.shape[1] == ndim:
        src = np.concatenate([src, np.ones((src.shape[0], 1))], axis=1)

    return src


########################################################################################################################

def rotation_matrix(
        ndim: int,
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
    ndim : int
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

    if ndim not in {2, 3}:
        raise ValueError('nd can be 2 or 3')

    if not ((ndim == 2 and np.isscalar(angle)) or (ndim == 3 and (np.isscalar(angle) or len(angle) == 3))):
        raise ValueError(
            'for 2d rotation angle need to be scalar, and for 3d rotation angle need to be a tuple with size of 3 or scalar')

    if degree:
        angle = np.deg2rad(angle)

    if ndim == 2:
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


def translation_matrix(
        ndim: int,
        translation: float | tuple,
        axis: int | tuple | None = None
) -> np.ndarray:
    if axis is None:
        axis = tuple(range(ndim))
    if np.isscalar(axis):
        axis = (axis,)

    if not all(a < ndim for a in axis):
        raise ValueError('axis is out of range for the given ndim')

    if np.isscalar(translation):
        translation = tuple(translation if a in axis else 0 for a in range(ndim))
    elif len(translation) != len(axis):
        raise ValueError('translation and axis has different size')
    else:
        t_fix = [0] * ndim
        t_iter = iter(translation)
        for i, ax in enumerate(axis):
            t_fix[ax] = next(t_iter)
        translation = t_fix

    out = np.eye(ndim + 1, dtype=np.float64)
    out[:ndim, ndim] = np.asarray(translation, dtype=np.float64)
    return out


def affine_matrix(
        ndim: int = 2,
        rotation: float | tuple | None = None,
        translation: float | tuple | None = None,
        scale: float | tuple | None = None,
        shear: float | tuple | None = None,
) -> np.ndarray:
    assert ndim == 2

    if rotation is None:
        rotation = (0, 0)
    elif np.isscalar(rotation):
        rotation = (rotation, rotation)
    elif len(rotation) != ndim:
        raise ValueError('rotation size need to be equal to ndim')

    if translation is None:
        translation = (0, 0)
    elif np.isscalar(translation):
        translation = (translation, translation)
    elif len(translation) != ndim:
        raise ValueError('translation size need to be equal to ndim')

    if scale is None:
        scale = (1, 1)
    elif np.isscalar(scale):
        scale = (scale, scale)
    elif len(scale) != ndim:
        raise ValueError('scale size need to be equal to ndim')

    if shear is None:
        shear = (0, 0)
    elif np.isscalar(shear):
        shear = (shear, 0)
    elif len(shear) != ndim:
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


########################################################################################################################

class ProjectiveTransform(_BaseGeometricTransform):
    """
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

    def __init__(self, ndim: int = 2, matrix: np.ndarray | None = None):
        self.matrix = _valid_matrix(ndim, matrix)

    def __repr__(self):
        _out = f"{self.__class__.__name__}: ndim={self.ndim}, " \
               f"matrix = \n{np.array2string(self.matrix, separator=', ')}"
        return _out

    def __call__(self, src: np.ndarray) -> np.ndarray:
        src = valid_src(src, self.ndim)
        return self._transform(src, self.matrix)

    def __array__(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float64)

    def _transform(self, src: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        nd = matrix.shape[0] - 1

        dst = src @ matrix.T
        dst[dst[:, nd] == 0, nd] = np.finfo(float).eps
        dst[:, :nd] /= dst[:, nd:nd + 1]
        return dst

    @property
    def ndim(self) -> int:
        return self.matrix.shape[0] - 1

    @property
    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)

    def transform(self, src: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        matrix = _valid_matrix(self.ndim, matrix)
        src = valid_src(src, matrix.shape[0] - 1)
        return self._transform(src, matrix)


class RidgeTransform(ProjectiveTransform):
    """
    Ridge Transformation - Translate, Rotate.

    Ridge transformation includes rotation, and transformation.
    It preserves distances between points and angles, making it a rigid transformation.
    In other words, Ridge transformations maintain the shape and size of objects in the transformed space:

        | cos(theta), sin(theta), tx |
        | sin(theta), cos(theta), ty |
        |     0     ,      0    ,  1 |

    where theta is the rotation angle and (tx, ty) is the translation vector.
    """

    def __init__(
            self,
            ndim: int = 2,
            rotation: float | tuple | None = None,
            translation: float | tuple | None = None,
            matrix: np.ndarray | None = None,
    ):
        super().__init__(ndim=ndim, matrix=matrix)
        if matrix is None:
            if rotation is not None:
                if ndim not in {2, 3}:
                    raise ValueError('rotation parameter only supported for 2D and 3D.')
                self.matrix[:ndim, :ndim] = rotation_matrix(ndim, rotation, degree=True)
            if translation is not None:
                self.matrix[:ndim, ndim] = translation_matrix(ndim, translation)[:ndim, ndim]

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:self.ndim, self.ndim]

    @property
    def rotation(self) -> np.ndarray:
        if self.ndim == 2:
            return np.array(np.rad2deg(np.arctan2(self.matrix[1, 0], self.matrix[1, 1])))
        elif self.ndim == 3:
            return self.matrix[:3, :3]
        else:
            raise NotImplementedError('rotation parameter implemented only for 2D and 3D')


class SimilarityTransform(RidgeTransform):
    """
    Similarity transformation - Scale (uniform), Translate, Rotate.

    A similarity transformation, also known as a scaling and rotation transformation,
    preserves the shape of an object while allowing for uniform scaling and rotation.
    It includes translation, rotation, and scaling components.
    The transformation matrix for 2D similarity transformations looks like:

        | s*cos(theta), -s*sin(theta), tx |
        | s*sin(theta),  s*cos(theta), ty |
        |     0       ,     0       ,  1  |

    where s is the scaling factor, theta is the rotation angle and (tx, ty) is the translation vector.
    """

    def __init__(
            self,
            ndim: int = 2,
            rotation: float | tuple | None = None,
            translation: float | tuple | None = None,
            scale: float | None = None,
            matrix: np.ndarray | None = None,
    ):
        super().__init__(ndim=ndim, rotation=rotation, translation=translation, matrix=matrix)
        if matrix is None and scale is not None:
            self.matrix[:ndim, :ndim] *= scale

    @property
    def scale(self) -> np.ndarray:
        return np.sqrt(np.sum(self.matrix ** 2, axis=0))[:self.ndim]


class AffineTransform(ProjectiveTransform):
    """
    Affine transformation - Scale, Shear, Translate, Rotate.

    An affine transformation includes scaling, rotation, translation, and shear.
    It preserves straight lines and parallelism.
    The transformation matrix for 2D affine transformations is:

        | a, b, tx |       | a0, a1, a2 |
        | c, d, ty |       | b0, b1, b2 |
        | 0, 0, 1  |       |  0,  0,  1 |

    where a, b, c, d are scaling, rotation, and shear parameters and (tx, ty) is the translation vector.
    """

    def __init__(
            self,
            ndim: int = 2,
            rotation: float | tuple | None = None,
            translation: float | tuple | None = None,
            scale: float | tuple | None = None,
            shear: float | tuple | None = None,
            matrix: np.ndarray | None = None
    ):
        super().__init__(ndim=ndim, matrix=matrix)
        if matrix is None and any(p is not None for p in (rotation, translation, scale, shear)):
            if ndim != 2:
                raise ValueError('transform parameter only supported for 2D.')
            self.matrix = affine_matrix(
                ndim=self.ndim, rotation=rotation, translation=translation, scale=scale, shear=shear
            )

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:self.ndim, self.ndim]


########################################################################################################################
