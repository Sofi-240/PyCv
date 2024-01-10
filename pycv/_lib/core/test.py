import numpy as np
from pycv._lib.core import ops

def test_convolve(image: np.ndarray, kernel: np.ndarray) -> bool:
    array_shape = image.shape
    output = np.zeros(array_shape, image.dtype)

    center = tuple(s // 2 for s in kernel.shape)
    border_weight = tuple((c, int(s - c - 1)) for s, c in zip(kernel.shape, center))

    image = np.pad(image, tuple((c, c) for c in center), mode='reflect')

    ops.convolve(image, kernel, output)

    border_mask = np.zeros(image.shape, dtype=bool)
    for i, (b1, b2) in enumerate(border_weight):
        np.moveaxis(border_mask, i, 0)[:b1] = 1
        np.moveaxis(border_mask, i, 0)[-b2:] = 1
    border_mask = border_mask.ravel()

    unraveled_neighborhood = np.stack([(idx - c) for idx, c in zip(np.nonzero(kernel), center)], axis=-1)
    offsets_unraveled = unraveled_neighborhood[:, ::-1]

    image_shape = image.shape[::-1]

    jump = image_shape[1:] + (1,)
    jump = np.cumprod(jump[::-1])[::-1]
    raveled_neighborhood = (offsets_unraveled * jump).sum(axis=1)

    output_test = np.zeros(array_shape, image.dtype)
    output_test_ravel = output_test.ravel()
    image_ravel = image.ravel()
    kernel_ravel = kernel[kernel != 0]
    jj = 0

    for i in range(image.size):
        if not border_mask[i]:
            for k in range(len(raveled_neighborhood)):
                output_test_ravel[jj] += image_ravel[i - raveled_neighborhood[k]] * kernel_ravel[k]
            jj += 1

    if output.dtype.kind == 'f':
        output = np.round(output, 4)
        output_test = np.round(output_test, 4)

    return np.all(output == output_test)