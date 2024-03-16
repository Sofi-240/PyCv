def transform_hough_circle_peak():
    import numpy as np
    from pycv.draw import draw_circle
    from pycv.transform import hough_circle_peak, hough_circle
    from pycv.io import show_collection
    from pycv.morphological import region_fill, binary_dilation, binary_edge

    circles = np.zeros((101, 150), bool)

    for c1, c2, r in zip((51, 40, 55), (31, 80, 120), (30, 15, 20)):
        draw_circle((c1, c2), r, circles)
        region_fill(circles, (c1, c2), inplace=True)

    circles_edge = binary_edge(circles, 'outer')
    radius = np.arange(10, 35, 5)
    h_space = hough_circle(circles_edge, radius)

    peaks_h, peaks_radius, peaks_center = hough_circle_peak(h_space, radius)

    detected_circles = np.zeros_like(circles)

    for i in range(peaks_center.shape[0]):
        draw_circle(tuple(int(c) for c in peaks_center[i]), int(peaks_radius[i]), detected_circles)

    detected_circles = binary_dilation(detected_circles)

    marked = np.zeros(detected_circles.shape + (3,), bool)
    marked[..., 0] = circles | detected_circles
    marked[..., 1] = circles & ~detected_circles
    marked[..., 2] = marked[..., 1]

    marked = marked.astype(np.uint8) * 255

    show_collection([circles, marked])
