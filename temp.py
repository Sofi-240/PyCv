import numpy as np
import os.path as osp
from pycv.io import ImageLoader, show_collection, show_histograms
from pycv._lib.misc.histogram import histogram
from pycv._lib._src import c_pycv
from pycv.filters import hist_equalize, adjust_exp, adjust_log, adjust_gamma, adjust_linear
from pycv._lib.array_api.dtypes import cast

########################################################################################################################

def plot_adjusts(_images, _titles=None):
    _his = [histogram(_img, bins=256) for _img in _images]
    show_histograms(
        _images,
        [_h.normalize[0] for _h in _his],
        [_h.bins for _h in _his],
        [_h.cdf()[0] for _h in _his],
        titles=_titles
    )


loader = ImageLoader(osp.join(osp.dirname(__file__), '_debug_utils', 'data'))

moon_in = loader.load('moon')
# moon_in = cast(moon_in, np.float64)

gamma_correction = adjust_gamma(moon_in, gamma=2)
log_correction = adjust_log(moon_in)
exp_correction = adjust_exp(moon_in)

plot_adjusts(
    [moon_in, gamma_correction, log_correction, exp_correction],
    ['orig.', 'gamma correction', 'log correction', 'exp correction']
)

vl, vh = np.percentile(moon_in, [2, 98])
linear_correction = adjust_linear(moon_in, in_range=(vl, vh), out_range=(0, 1))
equalization = hist_equalize(moon_in, n_bins=256)

plot_adjusts(
    [moon_in, linear_correction, equalization],
    ['orig.', 'linear correction', 'equalization']
)
