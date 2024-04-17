import numpy as np
from pycv.draw import mark_points
from pycv.measurements import NRegionProperties
from pycv.io import ImageLoader, show_collection, DEFAULT_DATA_PATH
from pycv.segmentation import Thresholds, im_threshold
from pycv.filters import hist_equalize
from pycv.morphological import im_label, gray_opening, Strel, remove_small_objects, remove_small_holes, binary_erosion


########################################################################################################################

loader = ImageLoader(DEFAULT_DATA_PATH)
coins = loader.load('coins')

equalize = hist_equalize(coins, n_bins=256)

coins_open = gray_opening(equalize, Strel.CIRCLE(3))
coins_bin, th = im_threshold(coins_open, Thresholds.OTSU)

fig = show_collection([coins, equalize, coins_open, coins_bin])[0]
fig.axes[0].set_title('coins orig.')
fig.axes[1].set_title('equalization')
fig.axes[2].set_title('coins gray opening')
fig.axes[3].set_title('Otsu threshold')

ero = binary_erosion(coins_bin, Strel.SQUARE(5))
ero_rem = remove_small_objects(ero, threshold=96)
circles = remove_small_holes(ero_rem, threshold=64, connectivity=2)

fig = show_collection([coins_bin, ero, ero_rem, circles])[0]
fig.axes[0].set_title('coins bin')
fig.axes[1].set_title('binary erosion')
fig.axes[2].set_title('after removing small objects')
fig.axes[3].set_title('after removing small holes')

n_labels, labels = im_label(circles)

props = NRegionProperties()
props(labels)

center = [(cent + 0.5).astype(np.int64) for cent in props.centroid[:]]

marked = mark_points(circles, center, "CIRCLE", (255, 0, 0))

fig = show_collection([circles, marked])[0]
fig.axes[0].set_title('coins')
fig.axes[1].set_title('coins center')

########################################################################################################################
