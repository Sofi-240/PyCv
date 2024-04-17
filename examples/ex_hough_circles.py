import numpy as np
from pycv.draw import draw_circle
from pycv.io import ImageLoader, show_collection, DEFAULT_DATA_PATH
from pycv.segmentation import Thresholds, im_threshold
from pycv.morphological import gray_opening, Strel, binary_edge, binary_dilation, remove_small_objects, \
    remove_small_holes
from pycv.transform import hough_circle_peak, hough_circle


########################################################################################################################

def _prep_coins_(_image) -> np.ndarray:
    _open = gray_opening(_image, Strel.CIRCLE(2))
    _bin, _ = im_threshold(_image, Thresholds.OTSU)
    _bin = remove_small_objects(_bin, threshold=64)
    _bin = remove_small_holes(_bin, threshold=96, connectivity=2)
    _bin = binary_dilation(_bin)
    return _bin


########################################################################################################################

loader = ImageLoader(DEFAULT_DATA_PATH)
coins = loader.load('coins')[80:180, 300:-50]

coins_bin = _prep_coins_(coins)
edge = binary_edge(coins_bin)

fig = show_collection([coins_bin, edge])[0]
fig.axes[0].set_title('coins bin.')
fig.axes[1].set_title('coins edge.')

radius = np.arange(35, 50, 2)
h_space = hough_circle(edge, radius)

peaks_h, peaks_radius, peaks_center = hough_circle_peak(h_space, radius, n_peaks=3)

detected_circles = np.zeros_like(coins_bin)

for i in range(peaks_center.shape[0]):
    draw_circle(tuple(int(c) for c in peaks_center[i]), int(peaks_radius[i]), detected_circles)

detected_circles = binary_dilation(detected_circles)

marked = np.zeros(detected_circles.shape + (3,), bool)
marked[..., 0] = coins_bin | detected_circles
marked[..., 1] = coins_bin & ~detected_circles
marked[..., 2] = marked[..., 1]

marked = marked.astype(np.uint8) * 255

show_collection([coins_bin, marked])

########################################################################################################################
