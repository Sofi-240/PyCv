import numpy as np
from ._features import structure_tensor

__all__ = [
    'harris_corner'
]


########################################################################################################################

def harris_corner(
        image: np.ndarray,
        nobel_measure: bool = False,
        k: float = 0.05,
        epsilon: float = 1e-5,
        sigma: float | tuple = 1.
) -> np.ndarray:
    tensor = structure_tensor(image, sigma=sigma, padding_mode='constant', constant_value=0.)

    det = tensor[0, 0] * tensor[1, 1] - tensor[0, 1] ** 2
    trace = tensor[0, 0] + tensor[1, 1]

    if not nobel_measure:
        R = det - k * trace ** 2
    else:
        R = 2 * det / (trace + epsilon)

    return R


########################################################################################################################
