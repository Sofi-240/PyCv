import numpy as np
from .._lib.array_api.dtypes import get_dtype_info

__all__ = [
    'imshow',
    'show_collection',
    'show_struct'
]


########################################################################################################################

def _get_fig(nrows, ncols):
    from matplotlib import pyplot as plt
    fig, _ = plt.subplots(nrows, ncols, subplot_kw={'xticks': [], 'yticks': []})
    return fig


def _get_ax():
    from matplotlib import pyplot as plt
    _, ax = plt.subplots(1, 1, subplot_kw={'xticks': [], 'yticks': []})
    return ax


def _show(image: np.ndarray, ax=None, color_bar: bool = False):
    image = np.asarray(image)
    ax = ax or _get_ax()
    i_min, i_max = np.amin(image), np.amax(image)
    info = get_dtype_info(image.dtype)
    if i_min < 0:
        v = max(abs(i_max), abs(i_max))
        cmap = 'RdBu'
        i_min, i_max = -v, v
    elif np.issubdtype(image.dtype, np.floating) and abs(i_min) > 1 or abs(i_max) > 1:
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


########################################################################################################################

def imshow(image: np.ndarray, ax=None, color_bar: bool = False):
    _show(image, ax, color_bar)


def show_collection(collection: list | np.ndarray, fig=None):
    n = collection.shape[0] if isinstance(collection, np.ndarray) else len(collection)

    nn = n // 6

    if nn > 1 and fig is not None:
        raise ValueError('maximum images per figure is 6')
    elif fig is not None:
        fig = [fig]
    else:
        fig = [_get_fig(2, 3) for _ in range(nn - 1)]

    n_split = n - nn * 6

    if n_split < 3:
        nrows, ncols = 1, n_split
    elif n_split == 4:
        nrows = ncols = 2
    else:
        nrows, ncols = 2, 3

    fig.append(_get_fig(nrows, ncols))

    i = 0
    for f in fig:
        for ax in f.axes:
            _show(collection[i], ax=ax)
            i += 1
            if i == n:
                break


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

