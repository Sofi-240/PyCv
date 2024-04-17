
def harris_corner():
    import numpy as np
    import os.path as osp
    from pycv.io import ImageLoader, show_collection
    from pycv.features import harris_corner, find_peaks
    from pycv.draw import mark_points

    loader = ImageLoader(osp.join(osp.dirname(__file__), '_debug_utils', 'data'))

    brick = loader.load('brick')
    c1 = harris_corner(brick)
    peaks = find_peaks(c1, min_distance=7, threshold=0.001)

    marked = mark_points(brick, np.stack(np.where(peaks), axis=-1))

    show_collection([brick, marked])