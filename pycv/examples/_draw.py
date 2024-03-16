
def draw_mark_points():
    import numpy as np
    from pycv.draw import mark_points, Shapes, draw_circle
    from pycv.measurements import Bbox
    from pycv.io import show_collection
    from pycv.morphological import im_label, find_object, region_fill

    circles = np.zeros((101, 150), bool)

    for c1, c2, r in zip((51, 40, 55), (31, 80, 120), (30, 15, 20)):
        draw_circle((c1, c2), r, circles)
        region_fill(circles, (c1, c2), inplace=True)

    n_labels, labels = im_label(circles)

    marked = mark_points(
        circles,
        [Bbox(bbox).centroid_point for bbox in find_object(labels, as_slice=True)],
        Shapes.CROSS,
        (255, 0, 0)
    )

    show_collection([circles, marked])