import numpy as np
from .._lib.array_api.dtypes import get_dtype_info
from pycv._lib.misc.histogram import histogram, Histogram

__all__ = [
    'get_ax',
    'get_fig',
    'imshow',
    'show_collection',
    'show_struct',
    'show_histogram',
    'show_histograms',
    'show_pyramid',
    'show_scale_space'
]


########################################################################################################################

def get_fig(nrows, ncols, ticks: bool = False):
    from matplotlib import pyplot as plt
    subplot_kw = {'xticks': [], 'yticks': []} if not ticks else {}
    fig, _ = plt.subplots(nrows, ncols, subplot_kw=subplot_kw)
    return fig


def get_ax(ticks: bool = False):
    from matplotlib import pyplot as plt
    subplot_kw = {'xticks': [], 'yticks': []} if not ticks else {}
    _, ax = plt.subplots(1, 1, subplot_kw=subplot_kw)
    return ax


def imshow(image: np.ndarray, ax=None, color_bar: bool = False):
    image = np.asarray(image)
    ax = ax or get_ax()
    i_min, i_max = np.amin(image), np.amax(image)
    info = get_dtype_info(image.dtype)
    if i_min < 0:
        v = max(abs(i_max), abs(i_max))
        cmap = 'RdBu'
        i_min, i_max = -v, v
    elif np.issubdtype(image.dtype, np.floating) and (abs(i_min) > 1 or abs(i_max) > 1):
        cmap = 'viridis'
    else:
        i_min = 0
        i_max = info.max_val
        cmap = 'gray'
    ax.imshow(image, cmap=cmap, vmin=i_min, vmax=i_max)
    if color_bar:
        import matplotlib as mpl
        fig = ax.get_figure()
        fig.colorbar(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin=i_min, vmax=i_max), cmap=cmap), ax=ax,
            orientation='vertical'
        )
    return ax


########################################################################################################################


def show_collection(collection: list | np.ndarray):
    n = collection.shape[0] if isinstance(collection, np.ndarray) else len(collection)

    nn = n // 6

    fig = [get_fig(2, 3) for _ in range(nn - 1)]

    n_split = n - nn * 6

    if n_split < 3:
        nrows, ncols = 1, n_split
    elif n_split <= 4:
        nrows = ncols = 2
    else:
        nrows, ncols = 2, 3

    fig.append(get_fig(nrows, ncols))

    i = 0
    for f in fig:
        for ax in f.axes:
            imshow(collection[i], ax=ax)
            i += 1
            if i == n:
                break
    return fig


def show_struct(struct: np.ndarray):
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(8, 8))
    if struct.ndim == 3:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.voxels(struct)
        return
    if struct.ndim == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(struct, cmap="Paired", vmin=0, vmax=12)
        return
    raise ValueError(f'array dimensions need to be 2 or 3')


########################################################################################################################

def show_histogram(
        hist: np.ndarray,
        bins: np.ndarray,
        cdf: np.ndarray | None = None,
        image: np.ndarray | None = None,
        ax_image: None = None,
        ax_hist: None = None,
        max_yticks: int = 5,
        max_xticks: int = 10,
):
    assert hist.ndim == bins.ndim == 1
    assert hist.size == bins.size

    if image is not None:
        if ax_image is None:
            if ax_hist is None:
                fig = get_fig(1, 2, True)
                ax_image = fig.axes[0]
                ax_hist = fig.axes[1]

        ax_image = ax_image or get_ax(True)
        imshow(image, ax=ax_image)
        ax_image.set_axis_off()

    ax_hist = ax_hist or get_ax(True)

    ax_hist.stem(bins, hist)
    ax_hist.ticklabel_format(axis='y', scilimits=(0, 0))

    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(min(bins[0], 0), bins[-1])
    ax_hist.set_xticks(np.linspace(min(bins[0], 0), bins[-1], min(bins.size, max_xticks)))

    _, y_max = ax_hist.get_ylim()
    ax_hist.set_ylim(0, y_max)
    ax_hist.set_yticks(np.linspace(0, y_max, max_yticks))
    ax_hist.set_ylabel(f'{"N" if y_max > 1 else "%N"} pixels')

    if cdf is not None:
        assert cdf.shape == hist.shape
        axes_cdf = ax_hist.twinx()

        axes_cdf.plot(bins, cdf, 'r')
        axes_cdf.set_ylabel('CDF')
        axes_cdf.set_yticks(np.linspace(0, 1, max_yticks))
    return ax_image, ax_hist


def show_histograms(
        collection: np.ndarray | list[np.ndarray],
        histograms: np.ndarray | list[np.ndarray] | None = None,
        bins: np.ndarray | list[np.ndarray] | None = None,
        cdf: np.ndarray | list[np.ndarray] | None = None,
        titles: list[str] | None = None,
        max_yticks: int = 5,
        max_xticks: int = 5,
):
    ni = collection.shape[0] if isinstance(collection, np.ndarray) else len(collection)

    if histograms is None:
        histograms = []
        bins = []
        cdf = []
        for i in range(ni):
            h = histogram(collection[i], bins=256)
            histograms.append(h.counts[0])
            bins.append(h.bins)
            cdf.append(h.cdf()[0])

    nh = histograms.shape[0] if isinstance(histograms, np.ndarray) else len(histograms)
    if ni != nh and ni > 1:
        raise ValueError('number of images in collection need to be equal to number of histograms or only 1 image')

    n = ni + nh

    nn = n // 8

    fig = [get_fig(2, 4) for _ in range(nn - 1)]

    n_split = n - (nn - 1) * 8 if nn != 0 else n

    if n_split <= 4 and ni != nh:
        nrows, ncols = 1, n_split
    else:
        nrows, ncols = 2, n_split // 2

    fig.append(get_fig(nrows, ncols))

    if ni != nh:
        imshow(collection[0], ax=fig[0].axes[0])

    min_h = min(np.min(h) for h in histograms)
    min_h = min(min_h, 0)
    max_h = max(np.max(h) for h in histograms)

    i = 0
    for f in fig:
        nf = len(f.axes) // 2 if ni == nh else len(f.axes)
        pl = 0
        if i == 0 and ni != nh:
            nf -= 1
            pl = 1
        for a in range(nf):
            fi, fh = (None, f.axes[a + pl]) if ni != nh else (f.axes[a], f.axes[a + nf])
            show_histogram(
                histograms[i],
                bins[i],
                cdf[i] if cdf else None,
                collection[i] if ni == nh else None,
                fi, fh,
                max_yticks, max_xticks
            )
            if titles and fi:
                fi.set_title(titles[i])
            elif titles:
                fh.set_title(titles[i])
            if fi and a == 0:
                f.axes[a + nf].set_ylim(min_h, max_h)
                f.axes[a + nf].set_yticks(np.linspace(min_h, max_h, max_yticks))
            elif fi:
                f.axes[a + nf].set_ylim(min_h, max_h)
                f.axes[a + nf].set_yticks([])
                f.axes[a + nf].set_ylabel(None)
            elif not a or a == (nf + pl) // 2:
                f.axes[a].set_ylim(min_h, max_h)
                f.axes[a].set_yticks(np.linspace(min_h, max_h, max_yticks))
            else:
                f.axes[a].set_ylim(min_h, max_h)
                f.axes[a].set_yticks([])
                f.axes[a].set_ylabel(None)

            i += 1
            if i == nh:
                break

    return fig[0] if len(fig) == 1 else fig


########################################################################################################################

def pyramid_image(pyramid: list[np.ndarray], one_row: bool = False, one_col: bool = False) -> np.ndarray:
    if not one_row and not one_col:
        max_rows = max(pyramid[0].shape[0], sum(vv.shape[0] for vv in pyramid[1:]))
        max_cols = pyramid[0].shape[1] + pyramid[1].shape[1]

        p_image = np.zeros((max_rows, max_cols), dtype=np.float64)

        p_image[:pyramid[0].shape[0], :pyramid[0].shape[1]] = pyramid[0]

        r = 0
        c = pyramid[0].shape[1]

        for vv in pyramid[1:]:
            p_image[r:r + vv.shape[0], c:c + vv.shape[1]] = vv
            r += vv.shape[0]
        return p_image
    elif one_row:
        max_cols = sum(v.shape[1] for v in pyramid)
        max_rows = pyramid[0].shape[0]

        p_image = np.zeros((max_rows, max_cols), dtype=np.float64)
        p_image[:, :pyramid[0].shape[1]] = pyramid[0]

        c = pyramid[0].shape[1]
        r = 0

        for i in range(1, len(pyramid)):
            vv = pyramid[i]
            r += (pyramid[i - 1].shape[0] - pyramid[i].shape[0]) // 2
            p_image[r:r + vv.shape[0], c:c + vv.shape[1]] = vv
            c += vv.shape[1]
        return p_image

    max_cols = pyramid[0].shape[1]
    max_rows = sum(v.shape[0] for v in pyramid)

    p_image = np.zeros((max_rows, max_cols), dtype=np.float64)
    p_image[:pyramid[0].shape[0], :] = pyramid[0]

    c = 0
    r = pyramid[0].shape[0]

    for i in range(1, len(pyramid)):
        vv = pyramid[i]
        c += (pyramid[i - 1].shape[1] - pyramid[i].shape[1]) // 2
        p_image[r:r + vv.shape[0], c:c + vv.shape[1]] = vv
        r += vv.shape[0]
    return p_image


def show_pyramid(pyramid: list[np.ndarray], title: str | None = None, one_row: bool = False, one_col: bool = False):
    im = pyramid_image(pyramid, one_row, one_col)
    ax = imshow(im)
    if title:
        ax.set_title(title)

    return ax


def show_scale_space(scale_space: list[np.ndarray], title: str | None = None):
    n_row = len(scale_space)
    n_col = scale_space[0].shape[0]

    max_rows = sum(ss.shape[1] for ss in scale_space)
    max_cols = scale_space[0].shape[1] * n_col

    p_image = np.zeros((max_rows, max_cols), dtype=np.float64)
    r = 0
    c_shift = 0

    for i in range(n_row):
        c = c_shift
        for j in range(n_col):
            vv = scale_space[i][j]
            p_image[r:r + vv.shape[0], c:c + vv.shape[1]] = vv
            c += vv.shape[1] + 2 * c_shift
        r += scale_space[i][0].shape[0]
        if i < n_row - 1:
            c_shift += (scale_space[i].shape[2] - scale_space[i + 1].shape[2]) // 2

    ax = imshow(p_image)
    if title:
        ax.set_title(title)
    return ax

########################################################################################################################
